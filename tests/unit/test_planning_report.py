from src.urbanrenewal.agent.report import build_planning_report


def test_build_report_extracts_location_and_policy():
    events = [
        {
            "tool_name": "geocode_tool",
            "status": "completed",
            "result": {
                "ok": True,
                "summary": "鞍山新村，WGS84(121.50, 31.28)，位于杨浦区。",
                "source": "amap_geocode",
                "data": {
                    "name": "鞍山新村",
                    "address": "上海市杨浦区鞍山新村",
                    "district": "杨浦区",
                    "city": "上海市",
                    "lon_wgs84": 121.5,
                    "lat_wgs84": 31.28,
                    "in_target_district": True,
                },
            },
        },
        {
            "tool_name": "policy_rag_tool",
            "status": "completed",
            "result": {
                "ok": True,
                "summary": "政策 RAG 返回 2 条去重片段。",
                "source": "policy_chroma_rag",
                "data": {
                    "queries": ["适老化 无障碍"],
                    "count": 2,
                    "citations_markdown": "依据《完整居住社区建设指南》...",
                    "low_confidence": False,
                },
            },
        },
    ]

    report = build_planning_report("answer", events)

    assert report.answer_markdown == "answer"
    assert report.locations[0]["address"] == "上海市杨浦区鞍山新村"
    assert report.map_layers[0].layer_id == "analysis_center"
    assert report.policy_evidence[0].data["count"] == 2


def test_build_report_records_tool_failures_as_uncertainty():
    report = build_planning_report(
        "answer",
        [
            {
                "tool_name": "policy_rag_tool",
                "status": "completed",
                "result": {
                    "ok": False,
                    "error": "vector db unavailable",
                    "summary": "failed",
                    "source": "policy_chroma_rag",
                    "data": None,
                },
            }
        ],
    )

    assert "policy_rag_tool 调用失败" in report.uncertainties[0]


def test_build_report_extracts_poi_layer_and_gap_cards():
    report = build_planning_report(
        "answer",
        [
            {
                "tool_name": "poi_query_tool",
                "status": "completed",
                "result": {
                    "ok": True,
                    "summary": "800m 范围内检索到 2 个相关 POI。",
                    "source": "local_poi_parquet",
                    "data": {
                        "total_count": 2,
                        "summary": [{"category_planning": "养老服务", "count": 1}],
                        "geojson": {
                            "type": "FeatureCollection",
                            "features": [
                                {
                                    "type": "Feature",
                                    "geometry": {"type": "Point", "coordinates": [121.5, 31.28]},
                                    "properties": {"name": "养老服务点"},
                                }
                            ],
                        },
                    },
                },
            },
            {
                "tool_name": "facility_gap_tool",
                "status": "completed",
                "result": {
                    "ok": True,
                    "summary": "elderly_friendly 场景下：缺失 1 类，不足 1 类。",
                    "source": "local_poi_parquet",
                    "data": {
                        "scenario": "elderly_friendly",
                        "missing": ["公厕"],
                        "insufficient": ["养老服务"],
                        "present": ["公交站"],
                    },
                },
            },
        ],
    )

    assert any(layer.layer_id == "poi" for layer in report.map_layers)
    assert report.issues[0].severity == "high"
    assert "公厕" in report.issues[0].title
    assert report.recommendations[0].priority == "high"


def test_build_report_adds_policy_citation():
    report = build_planning_report(
        "answer",
        [
            {
                "tool_name": "policy_rag_tool",
                "status": "completed",
                "result": {
                    "ok": True,
                    "summary": "政策 RAG 返回 1 条去重片段。",
                    "source": "policy_chroma_rag",
                    "data": {
                        "queries": ["社区生活圈"],
                        "count": 1,
                        "citations_markdown": "依据《上海市15分钟社区生活圈规划导则》...",
                        "low_confidence": False,
                    },
                },
            }
        ],
    )

    assert report.policy_citations[0].confidence == "medium"
    assert "15分钟" in report.policy_citations[0].excerpt_markdown
