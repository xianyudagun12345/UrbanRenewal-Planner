from src.urbanrenewal.utils.ttl_cache import TTLCache


def test_ttl_cache_get_set_clear():
    cache = TTLCache[int](ttl_seconds=60, max_items=2)
    cache.set("a", 1)

    assert cache.get("a") == 1
    cache.clear()
    assert cache.get("a") is None


def test_ttl_cache_evicts_when_full():
    cache = TTLCache[int](ttl_seconds=60, max_items=1)
    cache.set("a", 1)
    cache.set("b", 2)

    assert cache.get("b") == 2
