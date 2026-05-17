"""
POI 空间查询工具。

核心功能：
- query_poi_by_buffer：以 WGS84 坐标为中心，构建圆形 buffer，返回范围内 POI
- summarize_poi：按规划分类统计 POI 数量，诊断设施配置情况
- check_facility_gaps：对照场景配置，输出缺失或不足的设施类型

查询逻辑：
  1. 将中心点和 GeoDataFrame 投影到 UTM（EPSG:32651，上海所在带）
  2. 用米制 buffer 过滤 POI
  3. 结果保留 WGS84 坐标，方便后续输出 GeoJSON

使用：
    from src.urbanrenewal.tools.poi_query import query_poi_by_buffer, check_facility_gaps
    gdf = query_poi_by_buffer(lon=121.5059, lat=31.2727, radius_m=800)
    gaps = check_facility_gaps(gdf, scenario="elderly_friendly")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import geopandas as gpd
import pandas as pd
from shapely import wkb
from shapely.geometry import Point

from src.urbanrenewal.config import cfg

# 上海地区 UTM 投影（Zone 51N），用于米制 buffer 计算
_UTM_CRS = "EPSG:32651"
_WGS84_CRS = "EPSG:4326"


# ---------------------------------------------------------------------------
# 数据加载（进程级缓存，只读一次）
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_poi_gdf() -> gpd.GeoDataFrame:
    """加载 POI parquet 并构建 GeoDataFrame，进程内只执行一次。"""
    df = pd.read_parquet(cfg.poi_path)
    df["geometry"] = df["geometry"].apply(
        lambda x: wkb.loads(x) if isinstance(x, bytes) else x
    )
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=_WGS84_CRS)
    return gdf


# ---------------------------------------------------------------------------
# 主查询接口
# ---------------------------------------------------------------------------

def query_poi_by_buffer(
    lon: float,
    lat: float,
    radius_m: int = 800,
    categories: Optional[list[str]] = None,
    scenario: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    以 (lon, lat) WGS84 坐标为中心，构建圆形 buffer 查询 POI。

    Args:
        lon:        中心点经度（WGS84）
        lat:        中心点纬度（WGS84）
        radius_m:   查询半径（米），默认 800
        categories: 按 category_planning 过滤，None 则返回所有类别
        scenario:   场景名称（如 "elderly_friendly"），自动从配置读取对应类别列表；
                    与 categories 同时指定时取并集

    Returns:
        GeoDataFrame，CRS 为 WGS84，新增列 distance_m（到中心点的直线距离）
    """
    gdf = _load_poi_gdf()

    # 投影到 UTM 做米制 buffer
    center_wgs = gpd.GeoSeries([Point(lon, lat)], crs=_WGS84_CRS)
    center_utm = center_wgs.to_crs(_UTM_CRS).iloc[0]
    gdf_utm = gdf.to_crs(_UTM_CRS)

    buffer_geom = center_utm.buffer(radius_m)
    mask = gdf_utm.geometry.within(buffer_geom)
    result = gdf[mask].copy()

    # 计算距中心点的直线距离（米）
    result_utm = gdf_utm[mask].copy()
    result["distance_m"] = result_utm.geometry.distance(center_utm).round(1).values

    # 按类别过滤
    target_cats: set[str] = set()
    if categories:
        target_cats.update(categories)
    if scenario and scenario in cfg.scenarios:
        target_cats.update(cfg.scenario_poi_categories(scenario))
    if target_cats:
        result = result[result["category_planning"].isin(target_cats)]

    result = result.sort_values("distance_m").reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# 统计与诊断
# ---------------------------------------------------------------------------

def summarize_poi(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    按 category_planning 统计 POI 数量和平均距离。

    Returns:
        DataFrame，列：category_planning, count, avg_distance_m, min_distance_m
    """
    if gdf.empty:
        return pd.DataFrame(columns=["category_planning", "count", "avg_distance_m", "min_distance_m"])

    summary = (
        gdf.groupby("category_planning")
        .agg(
            count=("poi_id", "count"),
            avg_distance_m=("distance_m", "mean"),
            min_distance_m=("distance_m", "min"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )
    summary["avg_distance_m"] = summary["avg_distance_m"].round(0).astype(int)
    summary["min_distance_m"] = summary["min_distance_m"].round(0).astype(int)
    return summary


@dataclass
class FacilityGapReport:
    """设施缺口诊断报告。"""
    scenario: str
    radius_m: int
    present: list[str] = field(default_factory=list)      # 已有设施
    missing: list[str] = field(default_factory=list)      # 完全缺失
    insufficient: list[str] = field(default_factory=list) # 数量偏少（<2）

    def __str__(self) -> str:
        lines = [f"[{self.scenario}] {self.radius_m}m 范围设施诊断"]
        if self.missing:
            lines.append(f"  [缺失] {'、'.join(self.missing)}")
        if self.insufficient:
            lines.append(f"  [不足] {'、'.join(self.insufficient)}")
        if self.present:
            lines.append(f"  [已有] {'、'.join(self.present)}")
        return "\n".join(lines)


def check_facility_gaps(
    gdf: gpd.GeoDataFrame,
    scenario: str,
    insufficient_threshold: int = 2,
) -> FacilityGapReport:
    """
    对照场景所需 POI 类别，输出缺失或不足的设施类型。

    Args:
        gdf:                   query_poi_by_buffer 返回的结果
        scenario:              场景名，需在 config/project.yaml 的 scenarios 中定义
        insufficient_threshold: 少于此数量视为"不足"，默认 2

    Returns:
        FacilityGapReport
    """
    required = cfg.scenario_poi_categories(scenario)
    radius_m = int(gdf["distance_m"].max()) if not gdf.empty else 0
    counts = gdf["category_planning"].value_counts().to_dict() if not gdf.empty else {}

    report = FacilityGapReport(scenario=scenario, radius_m=radius_m)
    for cat in required:
        cnt = counts.get(cat, 0)
        if cnt == 0:
            report.missing.append(cat)
        elif cnt < insufficient_threshold:
            report.insufficient.append(cat)
        else:
            report.present.append(cat)
    return report


# ---------------------------------------------------------------------------
# GeoJSON 输出
# ---------------------------------------------------------------------------

def to_geojson(gdf: gpd.GeoDataFrame, max_features: int = 500) -> dict:
    """
    将查询结果转为 GeoJSON dict，保留关键字段，控制输出体积。

    Args:
        gdf:          query_poi_by_buffer 返回的结果
        max_features: 最多输出条数，按距离截断
    """
    keep_cols = [
        "poi_id", "name", "category_planning", "category_big",
        "distance_m", "lon_wgs84", "lat_wgs84", "geometry",
    ]
    cols = [c for c in keep_cols if c in gdf.columns]
    subset = gdf[cols].head(max_features)
    return subset.__geo_interface__
