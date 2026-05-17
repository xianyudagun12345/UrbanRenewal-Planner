"""
地理编码工具：调用高德地图 API 将地名解析为 WGS84 坐标。

核心功能：
- geocode(place)：返回单个最优结果（优先杨浦区匹配）
- geocode_all(place)：返回所有候选结果

坐标转换：高德返回 GCJ-02，统一转换为 WGS84 供空间分析使用。

使用：
    from src.urbanrenewal.tools.geocode import geocode
    result = geocode("鞍山新村")
    # result.lon_wgs84, result.lat_wgs84
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import requests

from src.urbanrenewal.config import cfg

# GCJ-02 → WGS84 转换常量
_A = 6378245.0
_EE = 0.00669342162296594323


def _transform_lat(x: float, y: float) -> float:
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(y / 12.0 * math.pi) + 320 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
    return ret


def _transform_lon(x: float, y: float) -> float:
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(x / 12.0 * math.pi) + 300.0 * math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
    return ret


def gcj02_to_wgs84(lon: float, lat: float) -> tuple[float, float]:
    """将高德 GCJ-02 坐标转换为 WGS84。"""
    d_lat = _transform_lat(lon - 105.0, lat - 35.0)
    d_lon = _transform_lon(lon - 105.0, lat - 35.0)
    rad_lat = lat / 180.0 * math.pi
    magic = math.sin(rad_lat)
    magic = 1 - _EE * magic * magic
    sqrt_magic = math.sqrt(magic)
    d_lat = (d_lat * 180.0) / ((_A * (1 - _EE)) / (magic * sqrt_magic) * math.pi)
    d_lon = (d_lon * 180.0) / (_A / sqrt_magic * math.cos(rad_lat) * math.pi)
    return lon - d_lon, lat - d_lat


@dataclass
class GeocodeResult:
    name: str
    address: str
    district: str
    city: str
    lon_gcj02: float
    lat_gcj02: float
    lon_wgs84: float
    lat_wgs84: float
    in_target_district: bool

    def __str__(self) -> str:
        flag = "[杨浦]" if self.in_target_district else "[区外]"
        return (
            f"{flag} {self.name} | {self.address} | "
            f"WGS84({self.lon_wgs84:.6f}, {self.lat_wgs84:.6f})"
        )


def _parse_candidate(geo: dict) -> GeocodeResult:
    loc = geo.get("location", "0,0").split(",")
    lon_gcj, lat_gcj = float(loc[0]), float(loc[1])
    lon_wgs, lat_wgs = gcj02_to_wgs84(lon_gcj, lat_gcj)

    district = geo.get("district", "") or ""
    city_raw = geo.get("city", "") or ""
    province = geo.get("province", "") or ""
    city = city_raw if city_raw else province

    target_district = cfg.district
    in_target = target_district in district

    return GeocodeResult(
        name=geo.get("formatted_address", ""),
        address=geo.get("formatted_address", ""),
        district=district,
        city=city,
        lon_gcj02=lon_gcj,
        lat_gcj02=lat_gcj,
        lon_wgs84=round(lon_wgs, 7),
        lat_wgs84=round(lat_wgs, 7),
        in_target_district=in_target,
    )


def geocode_all(place: str, city: str = "上海市") -> list[GeocodeResult]:
    """查询地名的所有候选结果。"""
    url = "https://restapi.amap.com/v3/geocode/geo"
    params = {
        "key": cfg.amap_api_key,
        "address": place,
        "city": city,
        "output": "json",
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "1" or not data.get("geocodes"):
        return []

    return [_parse_candidate(g) for g in data["geocodes"]]


def geocode(
    place: str,
    city: str = "上海市",
    require_district: bool = True,
) -> Optional[GeocodeResult]:
    """
    解析地名，返回最优单个结果。

    优先级：
    1. 位于杨浦区的结果
    2. 若首次无结果，自动用"上海+地名"重试一次
    3. 若仍无杨浦区结果，返回第一个候选（调用方通过 in_target_district 判断）
    4. 若无任何结果，返回 None
    """
    candidates = geocode_all(place, city)

    if not candidates and not place.startswith("上海"):
        candidates = geocode_all("上海" + place, city)

    if not candidates:
        return None

    for c in candidates:
        if c.in_target_district:
            return c

    return candidates[0]
