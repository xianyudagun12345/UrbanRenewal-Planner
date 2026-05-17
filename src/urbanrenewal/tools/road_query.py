"""
路网查询工具。

核心功能：
- find_nearest_node:   给定 WGS84 坐标，返回最近路网节点
- query_isochrone:     步行/骑行等时圈（返回范围内节点 + 凸包多边形）
- query_shortest_path: 两点间最短步行路径（返回路径节点坐标 + 统计信息）
- query_intersections: 查询范围内路口节点
- query_edges_in_buffer: 查询范围内路段

数据说明：
  - 图文件：walk_bike_network.graphml（MultiDiGraph，节点带 x/y/WGS84 坐标）
  - 边权重：walk_time_min（步行分钟数），length_m（米）
  - 节点字段：x, y, osmid, street_count, degree_calc, is_intersection
  - 边字段：edge_id, length_m, walkable, bikeable, walk_time_min, road_name

使用：
    from src.urbanrenewal.tools.road_query import query_isochrone, query_shortest_path
    iso = query_isochrone(lon=121.5059, lat=31.2727, minutes=10)
    path = query_shortest_path(121.5059, 31.2727, 121.512, 31.280)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import networkx as nx
import numpy as np
from shapely.geometry import MultiPoint, Polygon

from src.urbanrenewal.config import cfg

_WALK_WEIGHT = "walk_time_min"
_BIKE_WEIGHT = "bike_time_min"
_METERS_PER_DEG_LAT = 111320.0


# ---------------------------------------------------------------------------
# 图加载（进程级缓存）
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_graph() -> nx.MultiDiGraph:
    """加载步行路网 GraphML，进程内只执行一次。"""
    return nx.read_graphml(str(cfg.road_graph_path))


@lru_cache(maxsize=1)
def _node_arrays() -> tuple[list[str], np.ndarray]:
    """返回 (node_id_list, coords_array[N,2])，coords 列为 (lon, lat)。"""
    G = _load_graph()
    node_ids = list(G.nodes())
    coords = np.array([[float(G.nodes[n]["x"]), float(G.nodes[n]["y"])] for n in node_ids])
    return node_ids, coords


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _haversine_approx(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """快速平面近似距离（米），杨浦区范围误差 < 0.1%。"""
    dx = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2)) * _METERS_PER_DEG_LAT
    dy = (lat2 - lat1) * _METERS_PER_DEG_LAT
    return math.sqrt(dx * dx + dy * dy)


def _nearest_node(lon: float, lat: float) -> tuple[str, float]:
    """返回 (node_id, distance_m)，使用向量化计算。"""
    node_ids, coords = _node_arrays()
    dx = (coords[:, 0] - lon) * math.cos(math.radians(lat)) * _METERS_PER_DEG_LAT
    dy = (coords[:, 1] - lat) * _METERS_PER_DEG_LAT
    dists = np.sqrt(dx * dx + dy * dy)
    idx = int(np.argmin(dists))
    return node_ids[idx], float(dists[idx])


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class NodeInfo:
    node_id: str
    lon: float
    lat: float
    street_count: int
    is_intersection: bool
    distance_m: float = 0.0  # 到查询中心点的距离


@dataclass
class IsochroneResult:
    center_lon: float
    center_lat: float
    minutes: float
    mode: str                          # "walk" | "bike"
    nodes: list[NodeInfo] = field(default_factory=list)
    hull_polygon: Optional[Polygon] = None   # 节点集合凸包

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    def to_geojson(self) -> dict:
        features = []
        if self.hull_polygon:
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [list(self.hull_polygon.exterior.coords)]},
                "properties": {"type": "isochrone", "minutes": self.minutes, "mode": self.mode},
            })
        for n in self.nodes:
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [n.lon, n.lat]},
                "properties": {
                    "node_id": n.node_id,
                    "is_intersection": n.is_intersection,
                    "street_count": n.street_count,
                    "distance_m": n.distance_m,
                },
            })
        return {"type": "FeatureCollection", "features": features}


@dataclass
class PathResult:
    origin_lon: float
    origin_lat: float
    dest_lon: float
    dest_lat: float
    walk_time_min: float
    length_m: float
    node_count: int
    coordinates: list[tuple[float, float]]  # LineString 坐标列表 [(lon, lat), ...]

    def to_geojson(self) -> dict:
        return {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [[c[0], c[1]] for c in self.coordinates]},
            "properties": {
                "walk_time_min": round(self.walk_time_min, 2),
                "length_m": round(self.length_m, 1),
                "node_count": self.node_count,
            },
        }


@dataclass
class IntersectionResult:
    lon: float
    lat: float
    node_id: str
    street_count: int
    degree_calc: int
    distance_m: float


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------

def find_nearest_node(lon: float, lat: float) -> NodeInfo:
    """返回距 (lon, lat) 最近的路网节点信息。"""
    G = _load_graph()
    node_id, dist = _nearest_node(lon, lat)
    d = G.nodes[node_id]
    return NodeInfo(
        node_id=node_id,
        lon=float(d["x"]),
        lat=float(d["y"]),
        street_count=int(d.get("street_count", 0)),
        is_intersection=str(d.get("is_intersection", "0")) == "1",
        distance_m=round(dist, 1),
    )


def query_isochrone(
    lon: float,
    lat: float,
    minutes: float = 10.0,
    mode: str = "walk",
) -> IsochroneResult:
    """
    计算步行/骑行等时圈。

    Args:
        lon, lat:  中心点 WGS84 坐标
        minutes:   时间阈值（分钟）
        mode:      "walk"（步行）或 "bike"（骑行）

    Returns:
        IsochroneResult，包含范围内节点列表和凸包多边形
    """
    G = _load_graph()
    weight = _WALK_WEIGHT if mode == "walk" else _BIKE_WEIGHT
    source_node, _ = _nearest_node(lon, lat)

    # ego_graph 返回从 source 出发在 radius 时间内可达的子图
    subgraph = nx.ego_graph(G, source_node, radius=minutes, distance=weight)

    nodes: list[NodeInfo] = []
    for n in subgraph.nodes():
        d = G.nodes[n]
        node_lon, node_lat = float(d["x"]), float(d["y"])
        dist = _haversine_approx(lon, lat, node_lon, node_lat)
        nodes.append(NodeInfo(
            node_id=n,
            lon=node_lon,
            lat=node_lat,
            street_count=int(d.get("street_count", 0)),
            is_intersection=str(d.get("is_intersection", "0")) == "1",
            distance_m=round(dist, 1),
        ))

    # 凸包多边形（至少 3 个节点才能计算）
    hull: Optional[Polygon] = None
    if len(nodes) >= 3:
        pts = MultiPoint([(n.lon, n.lat) for n in nodes])
        hull_geom = pts.convex_hull
        if isinstance(hull_geom, Polygon):
            hull = hull_geom

    return IsochroneResult(
        center_lon=lon,
        center_lat=lat,
        minutes=minutes,
        mode=mode,
        nodes=nodes,
        hull_polygon=hull,
    )


def query_shortest_path(
    origin_lon: float,
    origin_lat: float,
    dest_lon: float,
    dest_lat: float,
) -> Optional[PathResult]:
    """
    计算两点间步行最短路径。

    Args:
        origin_lon, origin_lat: 起点 WGS84
        dest_lon, dest_lat:     终点 WGS84

    Returns:
        PathResult，或 None（两点不连通）
    """
    G = _load_graph()
    src, _ = _nearest_node(origin_lon, origin_lat)
    dst, _ = _nearest_node(dest_lon, dest_lat)

    try:
        path_nodes = nx.shortest_path(G, src, dst, weight=_WALK_WEIGHT)
        walk_time = nx.shortest_path_length(G, src, dst, weight=_WALK_WEIGHT)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None

    # 累计路段长度
    total_length = 0.0
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        edge_data = min(G[u][v].values(), key=lambda d: float(d.get(_WALK_WEIGHT, 999)))
        total_length += float(edge_data.get("length_m", 0))

    coords = [(float(G.nodes[n]["x"]), float(G.nodes[n]["y"])) for n in path_nodes]

    return PathResult(
        origin_lon=origin_lon,
        origin_lat=origin_lat,
        dest_lon=dest_lon,
        dest_lat=dest_lat,
        walk_time_min=round(walk_time, 2),
        length_m=round(total_length, 1),
        node_count=len(path_nodes),
        coordinates=coords,
    )


def query_intersections(
    lon: float,
    lat: float,
    radius_m: float = 800.0,
    min_street_count: int = 3,
) -> list[IntersectionResult]:
    """
    查询范围内的路口节点。

    Args:
        lon, lat:          中心点 WGS84
        radius_m:          查询半径（米）
        min_street_count:  最少连接道路数，默认 3（过滤简单转折点）

    Returns:
        按距离排序的 IntersectionResult 列表
    """
    G = _load_graph()
    results: list[IntersectionResult] = []
    for node_id, d in G.nodes(data=True):
        node_lon, node_lat = float(d["x"]), float(d["y"])
        dist = _haversine_approx(lon, lat, node_lon, node_lat)
        if dist > radius_m:
            continue
        sc = int(d.get("street_count", 0))
        if sc < min_street_count:
            continue
        results.append(IntersectionResult(
            lon=node_lon,
            lat=node_lat,
            node_id=node_id,
            street_count=sc,
            degree_calc=int(d.get("degree_calc", 0)),
            distance_m=round(dist, 1),
        ))
    results.sort(key=lambda r: r.distance_m)
    return results


def query_edges_in_buffer(
    lon: float,
    lat: float,
    radius_m: float = 500.0,
) -> list[dict]:
    """
    查询范围内所有路段，返回包含 GeoJSON LineString 的字典列表。

    Returns:
        list of dict，每条包含：edge_id, road_name, length_m,
        walk_time_min, geometry（LineString GeoJSON）
    """
    G = _load_graph()
    results: list[dict] = []

    seen_edges: set[str] = set()
    for u, v, data in G.edges(data=True):
        edge_id = data.get("edge_id", f"{u}_{v}")
        if edge_id in seen_edges:
            continue
        # 用端点中点做粗过滤
        mid_lon = (float(G.nodes[u]["x"]) + float(G.nodes[v]["x"])) / 2
        mid_lat = (float(G.nodes[u]["y"]) + float(G.nodes[v]["y"])) / 2
        if _haversine_approx(lon, lat, mid_lon, mid_lat) > radius_m * 1.5:
            continue
        # 精确判断：端点任一在范围内
        dist_u = _haversine_approx(lon, lat, float(G.nodes[u]["x"]), float(G.nodes[u]["y"]))
        dist_v = _haversine_approx(lon, lat, float(G.nodes[v]["x"]), float(G.nodes[v]["y"]))
        if min(dist_u, dist_v) > radius_m:
            continue

        seen_edges.add(edge_id)
        u_coord = (float(G.nodes[u]["x"]), float(G.nodes[u]["y"]))
        v_coord = (float(G.nodes[v]["x"]), float(G.nodes[v]["y"]))
        results.append({
            "edge_id": edge_id,
            "road_name": data.get("road_name", "unknown"),
            "length_m": round(float(data.get("length_m", 0)), 1),
            "walk_time_min": round(float(data.get(_WALK_WEIGHT, 0)), 3),
            "geometry": {
                "type": "LineString",
                "coordinates": [list(u_coord), list(v_coord)],
            },
        })

    results.sort(key=lambda r: r["length_m"])
    return results
