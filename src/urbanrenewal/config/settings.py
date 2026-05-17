"""
项目配置单例：从 config/project.yaml 和 .env 统一加载所有配置。
所有模块通过 `from src.urbanrenewal.config import cfg` 获取配置。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # src/urbanrenewal/config -> project root
_CONFIG_PATH = _PROJECT_ROOT / "config" / "project.yaml"


@lru_cache(maxsize=1)
def _load() -> dict[str, Any]:
    load_dotenv(_PROJECT_ROOT / ".env", override=False)
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


class _Cfg:
    """扁平化访问 project.yaml 各节，同时暴露环境变量中的 API Key。"""

    def __init__(self) -> None:
        self._data = _load()

    # ---------- project ----------
    @property
    def district(self) -> str:
        return self._data["project"]["district"]

    @property
    def city(self) -> str:
        return self._data["project"]["city"]

    @property
    def full_district(self) -> str:
        return self.city + self.district

    # ---------- paths ----------
    @property
    def poi_path(self) -> Path:
        return Path(self._data["paths"]["poi_path"])

    @property
    def road_nodes_path(self) -> Path:
        return Path(self._data["paths"]["road_nodes_path"])

    @property
    def road_edges_path(self) -> Path:
        return Path(self._data["paths"]["road_edges_path"])

    @property
    def road_graph_path(self) -> Path:
        return Path(self._data["paths"]["road_graph_path"])

    @property
    def streetview_metadata_path(self) -> Path:
        return Path(self._data["paths"]["streetview_metadata_path"])

    @property
    def streetview_analysis_cache_path(self) -> Path:
        return Path(self._data["paths"]["streetview_analysis_cache_path"])

    @property
    def streetview_image_dir(self) -> Path:
        return Path(self._data["paths"]["streetview_dir"])

    @property
    def policy_raw_dir(self) -> Path:
        return Path(self._data["paths"]["policy_raw_dir"])

    @property
    def policy_processed_dir(self) -> Path:
        return Path(self._data["paths"]["policy_processed_dir"])

    @property
    def policy_chunks_path(self) -> Path:
        return Path(self._data["paths"]["policy_chunks_path"])

    @property
    def policy_documents_path(self) -> Path:
        return Path(self._data["paths"]["policy_documents_path"])

    @property
    def policy_vector_db_dir(self) -> Path:
        return Path(self._data["paths"]["policy_vector_db_dir"])

    # ---------- rag ----------
    @property
    def rag_collection_name(self) -> str:
        return self._data["rag"]["collection_name"]

    @property
    def rag_embedding_model(self) -> str:
        return self._data["rag"]["embedding_model"]

    @property
    def rag_chunk_size(self) -> int:
        return self._data["rag"]["chunk_size"]

    @property
    def rag_chunk_overlap(self) -> int:
        return self._data["rag"]["chunk_overlap"]

    @property
    def rag_top_k(self) -> int:
        return self._data["rag"]["top_k"]

    # ---------- analysis ----------
    @property
    def default_radius_m(self) -> int:
        return self._data["analysis"]["default_radius_m"]

    @property
    def walk_speed_kmh(self) -> float:
        return self._data["analysis"]["walk_speed_kmh"]

    @property
    def default_streetview_limit(self) -> int:
        return self._data["analysis"]["default_streetview_limit"]

    # ---------- scenarios ----------
    @property
    def scenarios(self) -> dict[str, Any]:
        return self._data["scenarios"]

    def scenario_poi_categories(self, scenario: str) -> list[str]:
        return self._data["scenarios"][scenario]["poi_categories"]

    def scenario_policy_keywords(self, scenario: str) -> list[str]:
        return self._data["scenarios"][scenario]["policy_keywords"]

    # ---------- API keys (from .env) ----------
    @property
    def dashscope_api_key(self) -> str:
        key = os.getenv("DASHSCOPE_API_KEY", "")
        if not key:
            raise EnvironmentError("DASHSCOPE_API_KEY 未在 .env 中设置")
        return key

    @property
    def dashscope_base_url(self) -> str:
        return os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    @property
    def llm_model(self) -> str:
        # .env 中优先读 MODEL，兼容 LLM_MODEL，最终回退 qwen-plus
        return os.getenv("MODEL") or os.getenv("LLM_MODEL", "qwen-plus")

    @property
    def vl_model(self) -> str:
        # .env 中优先读 VISION_MODEL，兼容 VL_MODEL，最终回退 qwen-vl-plus
        return os.getenv("VISION_MODEL") or os.getenv("VL_MODEL", "qwen-vl-plus")

    @property
    def amap_api_key(self) -> str:
        key = os.getenv("AMAP_API_KEY", "")
        if not key:
            raise EnvironmentError("AMAP_API_KEY 未在 .env 中设置")
        return key


cfg = _Cfg()
