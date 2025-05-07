import pytest
from unittest.mock import AsyncMock, patch
from datetime import datetime, timezone
from main import MetricsCollector, RateLimiter

@pytest.fixture
async def metrics_collector():
    collector = MetricsCollector()
    return collector

@pytest.fixture
async def mock_redis():
    return AsyncMock()

@pytest.fixture
async def rate_limiter(mock_redis):
    limiter = RateLimiter(mock_redis)
    return limiter

@pytest.mark.asyncio
async def test_track_request(metrics_collector):
    await metrics_collector.track_request(True, 'test_api_key')
    # Check total requests
    assert metrics_collector.metrics['total_requests'] == 1
    assert metrics_collector.metrics['cache_hit'] == 1
    assert metrics_collector.metrics['requests_test_api_key'] == 1

    await metrics_collector.track_request(False, 'test_api_key')
    # Check cache miss
    assert metrics_collector.metrics['total_requests'] == 2
    assert metrics_collector.metrics['cache_miss'] == 1
    assert metrics_collector.metrics['requests_test_api_key'] == 2

@pytest.mark.asyncio
async def test_get_metrics(metrics_collector):
    await metrics_collector.track_request(True, 'test_api_key')
    metrics = await metrics_collector.get_metrics()
    assert metrics['total_requests'] == 1
    assert metrics['cache_hit'] == 1

@pytest.mark.asyncio
async def test_reset_metrics(metrics_collector):
    await metrics_collector.track_request(True, 'test_api_key')
    assert metrics_collector.metrics['total_requests'] == 1
    await metrics_collector.reset_metrics()
    assert metrics_collector.metrics['total_requests'] == 0

@pytest.mark.asyncio
async def test_check_rate_limit(rate_limiter):
    api_key = 'test_api_key'
    # Initial call should allow under limit
    allow = await rate_limiter.check_rate_limit(api_key)
    assert allow is True

    # Simulate hitting the request limit
    rate_limiter.redis.zcard = AsyncMock(return_value=101)
    allow = await rate_limiter.check_rate_limit(api_key)
    assert allow is False

    # Check for a call after resetting the limit
    rate_limiter.redis.zcard = AsyncMock(return_value=99)
    allow = await rate_limiter.check_rate_limit(api_key)
    assert allow is True