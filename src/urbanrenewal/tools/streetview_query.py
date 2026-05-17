"""
街景查询工具。

核心功能：
- query_streetview_by_buffer：以 WGS84 坐标为中心，返回范围内街景图片元数据
- analyze_streetview_image：对单张图片调用多模态大模型（Qwen-VL）按需分析，结果写入本地缓存
- query_and_analyze：组合接口，查询 + 按需分析，返回带分析结果的街景列表

设计原则（来自 claude.md 约束）：
- 不提前全量分析所有图片
- 用户提问后按需检索图片并分析
- 分析结果写入 image_analysis_cache.parquet，已分析过的图片直接读缓存，不重复调用 API

使用：
    from src.urbanrenewal.tools.streetview_query import query_and_analyze
    results = query_and_analyze(lon=121.5059, lat=31.2727, radius_m=300, scenario="walkability")
    for r in results:
        print(r.image_id, r.distance_m, r.analysis_text)
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from openai import OpenAI

from src.urbanrenewal.config import cfg

logger = logging.getLogger(__name__)

# 默认多模态模型，从 cfg.vl_model 读取（对应 .env 中 VISION_MODEL 或 VL_MODEL）
_DEFAULT_VL_MODEL = "qwen-vl-plus"

# 分析场景对应的提示词
_SCENE_PROMPTS: dict[str, str] = {
    "walkability": (
        "这是一张上海市杨浦区的街道街景图片（2019年拍摄）。"
        "请从步行友好与慢行交通角度分析图中可见的街道环境，重点关注："
        "①人行道宽度与连续性；②路面铺装质量与障碍物；③路口过街条件（斑马线、信号灯、等候区）；"
        "④非机动车道设置；⑤遮阴与街道绿化。"
        "输出：问题列表（每条以'-'开头，说明具体位置和问题）+ 总体步行友好评分（1-5分）。"
        "若图片不清晰或无法判断，请说明。"
    ),
    "elderly_friendly": (
        "这是一张上海市杨浦区的街道街景图片（2019年拍摄）。"
        "请从老年友好与适老化角度分析图中街道环境，重点关注："
        "①无障碍坡道与盲道；②人行道障碍物（停车、摊贩、施工）；③休憩设施（座椅、凉亭）；"
        "④照明条件；⑤过街安全设施；⑥地面铺装防滑情况。"
        "输出：问题列表（每条以'-'开头，说明具体位置和问题）+ 适老化评分（1-5分）。"
        "若图片不清晰或无法判断，请说明。"
    ),
    "life_circle": (
        "这是一张上海市杨浦区的街道街景图片（2019年拍摄）。"
        "请分析图中可见的街道空间与周边设施环境，重点关注："
        "①沿街底层功能业态（商业、服务、公共设施）；②公共空间品质；"
        "③街道活力与人流特征；④临近可见的公共设施（公交站、公厕、绿地入口等）。"
        "输出：可见设施与环境描述（简洁列举）+ 街道活力评分（1-5分）。"
        "若图片不清晰或无法判断，请说明。"
    ),
    "general": (
        "这是一张上海市杨浦区的街道街景图片（2019年拍摄）。"
        "请描述图中的街道空间环境，包括：道路类型与宽度、两侧建筑、人行道状况、"
        "绿化、设施和总体环境品质。用简短中文段落描述，不超过150字。"
    ),
}


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class StreetviewMeta:
    """单张街景图片的元数据。"""
    image_id: str
    image_abs_path: str
    lon_wgs84: float
    lat_wgs84: float
    direction_bucket: str   # N/NE/E/SE/S/SW/W/NW
    north_angle: float
    capture_ym: str         # 如 "2019-05"
    distance_m: float       # 到查询中心点的距离


@dataclass
class StreetviewAnalysis:
    """单张图片的分析结果（含元数据 + 多模态分析文本）。"""
    meta: StreetviewMeta
    analysis_text: Optional[str] = None   # 多模态分析结果；None 表示未分析
    scenario: Optional[str] = None
    model: Optional[str] = None
    analyzed_at: Optional[str] = None
    from_cache: bool = False

    # 便捷属性透传
    @property
    def image_id(self) -> str:
        return self.meta.image_id

    @property
    def distance_m(self) -> float:
        return self.meta.distance_m

    def to_dict(self) -> dict:
        return {
            "image_id": self.meta.image_id,
            "image_path": self.meta.image_abs_path,
            "lon_wgs84": self.meta.lon_wgs84,
            "lat_wgs84": self.meta.lat_wgs84,
            "direction": self.meta.direction_bucket,
            "capture_ym": self.meta.capture_ym,
            "distance_m": self.meta.distance_m,
            "analysis_text": self.analysis_text,
            "scenario": self.scenario,
            "from_cache": self.from_cache,
        }


# ---------------------------------------------------------------------------
# 元数据加载（进程级缓存）
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_metadata() -> pd.DataFrame:
    """加载街景 metadata，只保留 image_exists=True 的有效条目。"""
    df = pd.read_parquet(cfg.streetview_metadata_path)
    valid = df[df["image_exists"]].copy()
    valid = valid.reset_index(drop=True)
    logger.debug("街景 metadata 已加载：%d 张有效图片", len(valid))
    return valid


# ---------------------------------------------------------------------------
# 分析缓存读写
# ---------------------------------------------------------------------------

def _load_analysis_cache() -> dict[str, dict]:
    """
    从 image_analysis_cache.parquet 加载已有分析结果。
    key: image_id + "|" + scenario
    """
    cache_path = cfg.streetview_analysis_cache_path
    if not cache_path.exists():
        return {}
    try:
        df = pd.read_parquet(cache_path)
        cache: dict[str, dict] = {}
        for _, row in df.iterrows():
            key = f"{row['image_id']}|{row['scenario']}"
            cache[key] = {
                "analysis_text": row["analysis_text"],
                "model": row.get("model", ""),
                "analyzed_at": row.get("analyzed_at", ""),
            }
        return cache
    except Exception as e:
        logger.warning("读取分析缓存失败：%s", e)
        return {}


def _append_analysis_cache(records: list[dict]) -> None:
    """将新分析结果追加写入 image_analysis_cache.parquet。"""
    if not records:
        return
    cache_path = cfg.streetview_analysis_cache_path
    new_df = pd.DataFrame(records)
    if cache_path.exists():
        try:
            existing = pd.read_parquet(cache_path)
            combined = pd.concat([existing, new_df], ignore_index=True)
            # 去重：同一 image_id + scenario 保留最新
            combined = combined.drop_duplicates(
                subset=["image_id", "scenario"], keep="last"
            )
            combined.to_parquet(cache_path, index=False)
        except Exception as e:
            logger.warning("追加缓存失败，覆盖写入：%s", e)
            new_df.to_parquet(cache_path, index=False)
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        new_df.to_parquet(cache_path, index=False)
    logger.debug("分析缓存已更新：%d 条新记录", len(records))


# ---------------------------------------------------------------------------
# 多模态分析
# ---------------------------------------------------------------------------

def _encode_image(image_path: str) -> str:
    """读取图片文件并返回 base64 编码字符串。"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _call_vl_model(image_path: str, prompt: str, model: str) -> str:
    """
    调用 DashScope Qwen-VL 多模态模型分析单张图片。
    使用 OpenAI 兼容接口，图片以 base64 data URL 形式传入。
    """
    client = OpenAI(
        api_key=cfg.dashscope_api_key,
        base_url=cfg.dashscope_base_url,
    )
    b64 = _encode_image(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=600,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------

def query_streetview_by_buffer(
    lon: float,
    lat: float,
    radius_m: float = 500.0,
    limit: int = 8,
    direction: Optional[str] = None,
) -> list[StreetviewMeta]:
    """
    以 (lon, lat) WGS84 坐标为中心，返回范围内街景图片元数据。

    Args:
        lon, lat:   中心点 WGS84 坐标
        radius_m:   查询半径（米），默认 500
        limit:      最多返回图片数量（按距离排序），默认 8
        direction:  按朝向过滤，如 "N"/"SE"；None 则返回所有朝向

    Returns:
        按距离升序排列的 StreetviewMeta 列表
    """
    df = _load_metadata()
    lons = df["lon_wgs84"].values
    lats = df["lat_wgs84"].values
    cos_lat = np.cos(np.radians(lat))
    dx = (lons - lon) * cos_lat * 111320.0
    dy = (lats - lat) * 111320.0
    dists = np.sqrt(dx * dx + dy * dy)

    mask = dists <= radius_m
    if direction:
        mask &= (df["direction_bucket"].values == direction)

    subset = df[mask].copy()
    subset["distance_m"] = dists[mask]
    subset = subset.sort_values("distance_m").head(limit)

    results: list[StreetviewMeta] = []
    for _, row in subset.iterrows():
        results.append(StreetviewMeta(
            image_id=str(row["image_id"]),
            image_abs_path=str(row["image_abs_path"]),
            lon_wgs84=float(row["lon_wgs84"]),
            lat_wgs84=float(row["lat_wgs84"]),
            direction_bucket=str(row["direction_bucket"]),
            north_angle=float(row["north_angle"]),
            capture_ym=str(row.get("capture_ym", "")),
            distance_m=round(float(row["distance_m"]), 1),
        ))
    return results


def analyze_streetview_image(
    meta: StreetviewMeta,
    scenario: str = "general",
    model: Optional[str] = None,
    use_cache: bool = True,
) -> StreetviewAnalysis:
    """
    对单张街景图片进行多模态分析（按需调用，优先读缓存）。

    Args:
        meta:       StreetviewMeta 对象
        scenario:   分析场景（"walkability"/"elderly_friendly"/"life_circle"/"general"）
        model:      指定 VL 模型；None 时使用默认 qwen-vl-plus
        use_cache:  True 时先查本地缓存，命中则不调用 API

    Returns:
        StreetviewAnalysis（含分析文本）
    """
    vl_model = model or cfg.vl_model
    cache_key = f"{meta.image_id}|{scenario}"

    # 尝试从缓存读取
    if use_cache:
        cache = _load_analysis_cache()
        if cache_key in cache:
            hit = cache[cache_key]
            return StreetviewAnalysis(
                meta=meta,
                analysis_text=hit["analysis_text"],
                scenario=scenario,
                model=hit.get("model", vl_model),
                analyzed_at=hit.get("analyzed_at", ""),
                from_cache=True,
            )

    # 检查图片文件可访问性
    if not Path(meta.image_abs_path).exists():
        return StreetviewAnalysis(
            meta=meta,
            analysis_text=f"[图片文件不存在：{meta.image_abs_path}]",
            scenario=scenario,
            from_cache=False,
        )

    # 调用多模态 API
    prompt = _SCENE_PROMPTS.get(scenario, _SCENE_PROMPTS["general"])
    try:
        analysis_text = _call_vl_model(meta.image_abs_path, prompt, vl_model)
        logger.info("图片分析完成：%s [%s]", meta.image_id, scenario)
    except Exception as e:
        logger.error("图片分析失败 %s：%s", meta.image_id, e)
        analysis_text = f"[分析失败：{e}]"

    analyzed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 写入缓存
    _append_analysis_cache([{
        "image_id": meta.image_id,
        "scenario": scenario,
        "analysis_text": analysis_text,
        "model": vl_model,
        "analyzed_at": analyzed_at,
        "lon_wgs84": meta.lon_wgs84,
        "lat_wgs84": meta.lat_wgs84,
    }])

    return StreetviewAnalysis(
        meta=meta,
        analysis_text=analysis_text,
        scenario=scenario,
        model=vl_model,
        analyzed_at=analyzed_at,
        from_cache=False,
    )


def query_and_analyze(
    lon: float,
    lat: float,
    radius_m: float = 500.0,
    scenario: str = "general",
    limit: int = 5,
    model: Optional[str] = None,
    use_cache: bool = True,
) -> list[StreetviewAnalysis]:
    """
    组合接口：查询范围内街景 + 对每张图片按需多模态分析。

    Args:
        lon, lat:  中心点 WGS84 坐标
        radius_m:  查询半径（米）
        scenario:  分析场景
        limit:     最多分析图片数量（按距离选取最近的）
        model:     VL 模型；None 使用默认
        use_cache: 优先读本地缓存

    Returns:
        StreetviewAnalysis 列表，按距离升序
    """
    metas = query_streetview_by_buffer(lon, lat, radius_m=radius_m, limit=limit)
    if not metas:
        logger.info("查询范围内无有效街景图片（radius=%dm）", radius_m)
        return []

    results: list[StreetviewAnalysis] = []
    for meta in metas:
        analysis = analyze_streetview_image(meta, scenario=scenario, model=model, use_cache=use_cache)
        results.append(analysis)

    cached_count = sum(1 for r in results if r.from_cache)
    api_count = len(results) - cached_count
    logger.info(
        "街景分析完成：共 %d 张（缓存 %d，新调用 %d）",
        len(results), cached_count, api_count,
    )
    return results
