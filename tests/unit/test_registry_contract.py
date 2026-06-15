from src.urbanrenewal.tools.registry import TOOL_NAMES, _GEOCODE_CACHE, _POLICY_CACHE, _err, _jsonable, _ok


def test_registry_exposes_expected_tools():
    assert {
        "geocode_tool",
        "poi_query_tool",
        "facility_gap_tool",
        "isochrone_tool",
        "intersection_tool",
        "streetview_tool",
        "policy_rag_tool",
        "report_generation_tool",
    }.issubset(set(TOOL_NAMES))


def test_tool_result_contract_success():
    result = _ok(data={"count": 1}, summary="done", source="unit")
    assert result["ok"] is True
    assert result["data"] == {"count": 1}
    assert result["summary"] == "done"
    assert result["error"] == ""
    assert result["source"] == "unit"
    assert "cost_hint" in result


def test_tool_result_contract_error():
    result = _err(error="bad", source="unit")
    assert result["ok"] is False
    assert result["data"] is None
    assert result["error"] == "bad"
    assert result["source"] == "unit"


def test_jsonable_handles_objects():
    class Thing:
        def __str__(self):
            return "thing"

    assert _jsonable({"x": [Thing()]}) == {"x": ["thing"]}


def test_registry_caches_are_clearable():
    _GEOCODE_CACHE.set("unit", {"ok": True})
    _POLICY_CACHE.set("unit", {"ok": True})

    assert _GEOCODE_CACHE.get("unit") == {"ok": True}
    assert _POLICY_CACHE.get("unit") == {"ok": True}

    _GEOCODE_CACHE.clear()
    _POLICY_CACHE.clear()

    assert _GEOCODE_CACHE.get("unit") is None
    assert _POLICY_CACHE.get("unit") is None
