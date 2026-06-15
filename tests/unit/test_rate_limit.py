from src.urbanrenewal.api.rate_limit import InMemoryRateLimiter


def test_in_memory_rate_limiter_blocks_after_limit():
    limiter = InMemoryRateLimiter(max_requests=2, window_seconds=60)

    first = limiter.check("client")
    second = limiter.check("client")
    third = limiter.check("client")

    assert first.allowed is True
    assert second.allowed is True
    assert third.allowed is False
    assert third.retry_after_seconds > 0


def test_in_memory_rate_limiter_reset():
    limiter = InMemoryRateLimiter(max_requests=1, window_seconds=60)
    assert limiter.check("client").allowed is True
    assert limiter.check("client").allowed is False
    limiter.reset()
    assert limiter.check("client").allowed is True
