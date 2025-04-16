To create thorough unit and regression tests for the FastAPI application described in your `main.py` file, we can use the `pytest` framework along with `httpx` for testing API calls and `unittest.mock` for mocking external dependencies. Below is a structured approach to testing:

1. **Setting Up the Test Environment**
2. **Testing Configuration and Initialization**
3. **Testing the Rate Limiter**
4. **Testing Metrics Collector**
5. **Testing Redis Client**
6. **Testing API Endpoints**

Here's a complete test suite:

### Testing Structure using Pytest

Create a file named `test_main.py`.

```python
import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from main import app, Settings, RateLimiter, MetricsCollector, RedisClient, generate_cache_key

@pytest.fixture(scope="module")
def test_app():
    """Fixture to create a FastAPI test client."""
    client = TestClient(app)
    yield client


@pytest.fixture
async def mock_redis():
    """Mock Redis client for tests."""
    with patch("main.RedisClient") as mock:
        mock.return_value.get_client = AsyncMock(return_value=mock)
        yield mock


@pytest.fixture
async def rate_limiter(mock_redis):
    """Fixture for RateLimiter."""
    return RateLimiter(mock_redis)


@pytest.fixture
def metrics_collector():
    """Fixture for MetricsCollector."""
    return MetricsCollector()


@pytest.fixture
def sample_messages():
    """Fixture with sample messages."""
    return [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi!"}]


def test_generate_cache_key(sample_messages):
    """Test cache key generation for messages."""
    expected_length = 64  # Length of a SHA-256 hash
    cache_key = generate_cache_key(sample_messages)
    assert len(cache_key) == expected_length
    assert cache_key.startswith("cache:")


@pytest.mark.asyncio
async def test_rate_limiter_check(rate_limiter):
    """Verify that the rate limit check functions correctly."""
    api_key = "test_api_key"
    # Simulate rate-limited checks
    result = await rate_limiter.check_rate_limit(api_key)
    assert result is True  # Should allow if no requests have been made


@pytest.mark.asyncio
async def test_metrics_collection(metrics_collector):
    """Test metrics collection and tracking."""
    await metrics_collector.track_request(True, "test_api_key")
    metrics = await metrics_collector.get_metrics()
    
    assert metrics["total_requests"] == 1
    assert metrics["cache_hit"] == 1
    assert metrics["requests_test_api_key"] == 1

    await metrics_collector.reset_metrics()
    metrics = await metrics_collector.get_metrics()
    assert metrics["total_requests"] == 0


@pytest.mark.asyncio
async def test_health_endpoint(test_app, mock_redis):
    """Test health check endpoint."""
    mock_redis.return_value.ping.return_value = True
    response = test_app.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_cache_stats(test_app, mock_redis):
    """Test cache stats endpoint."""
    mock_redis.return_value.info.return_value = {
        "used_memory": 12345,
        "used_memory_human": "12.34K",
        "connected_clients": 5,
    }
    mock_redis.return_value.dbsize = AsyncMock(return_value=10)
    
    response = test_app.get("/cache/stats", headers={"X-API-Key": "valid_key"})
    assert response.status_code == 200
    assert response.json()["used_memory"] == 12345


@pytest.mark.asyncio
async def test_clear_cache(test_app, mock_redis):
    """Test the endpoint to clear cache."""
    mock_redis.return_value.flushdb = AsyncMock()
    response = test_app.delete("/cache", headers={"X-API-Key": "valid_key"})
    assert response.status_code == 200
    assert response.json() == {"status": "success", "message": "Cache cleared"}


# Error handling tests
@pytest.mark.asyncio
async def test_generic_exception_handler(test_app):
    """Test generic exception handling."""
    response = test_app.get("/nonexistent")
    assert response.status_code == 404  # As expected, since the endpoint doesn't exist


@pytest.mark.asyncio
async def test_valid_api_key(test_app):
    """Test valid API key is accepted."""
    response = test_app.get("/cache/stats", headers={"X-API-Key": "valid_key"})
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_invalid_api_key(test_app):
    """Test invalid API key is rejected."""
    response = test_app.get("/cache/stats", headers={"X-API-Key": "invalid_key"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API key"


if __name__ == '__main__':
    pytest.main()
```

### Explanation of Tests

1. **Test Client Initialization**: Sets up a test client using FastAPI's `TestClient`.
2. **Mock Dependencies**: Uses `unittest.mock` to mock Redis client and other dependencies for isolated unit testing.
3. **Generate Cache Key**: Tests if the cache keys are being generated correctly.
4. **Rate Limiter**: Validates the functionality of the RateLimiter class.
5. **Metrics Collection**: Verifies that metrics can be tracked and reset.
6. **Health Check**: Tests the `/health` endpoint for health status.
7. **Cache Stats**: Tests retrieval of cache statistics from Redis.
8. **Clear Cache**: Tests the endpoint that clears the cache.
9. **Error Handling**: Tests generic error handling functionality and valid/invalid API key scenarios.
10. **Async Testing**: Uses `pytest.mark.asyncio` to handle asynchronous function testing.

### Running the Tests

You can run the tests using the following command in your terminal:

```bash
pytest test_main.py
```

This setup is adaptable and can be expanded for additional edge cases or covered features as required by your application.