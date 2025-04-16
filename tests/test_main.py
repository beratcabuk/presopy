import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from main import app, redis_client, MetricsCollector

# Define a TestClient for the FastAPI app
client = TestClient(app)

@pytest.fixture
def mock_redis():
    with patch('main.RedisClient.get_client', new_callable=AsyncMock) as mock:
        yield mock

@pytest.fixture
def mock_metrics():
    with patch('main.MetricsCollector.get_metrics', new_callable=AsyncMock) as mock:
        yield mock

@pytest.fixture
def valid_api_key():
    return "valid_api_key"

@pytest.fixture(autouse=True)
def setup_environment(valid_api_key):
    # Set up environment variables and settings
    from main import settings
    settings.service_api_keys = [valid_api_key]


def test_health_check(mock_redis):
    """Verify health check returns healthy status and checks Redis connection."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_cache_stats(mock_redis, mock_metrics, valid_api_key):
    """Ensure cache stats endpoint returns correct information with valid API key."""
    mock_redis.client.info.return_value = {"used_memory": 12345, "used_memory_human": "12.3K", "connected_clients": 1}
    mock_redis.client.dbsize.return_value = 10
    mock_metrics.get_metrics.return_value = {"total_requests": 1}

    response = client.get("/cache/stats", headers={"X-API-Key": valid_api_key})
    assert response.status_code == 200
    data = response.json()
    assert data["used_memory"] == 12345
    assert data["total_keys"] == 10


def test_clear_cache(mock_redis, valid_api_key):
    """Validate cache clear operation returns success."""
    response = client.delete("/cache", headers={"X-API-Key": valid_api_key})
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_invalidate_cache_entry(mock_redis, valid_api_key):
    """Ensure specific cache entry invalidation works as expected."""
    cache_key = "test_key"
    response = client.delete(f"/cache/{cache_key}", headers={"X-API-Key": valid_api_key})
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_post_completions(mock_redis, valid_api_key):
    """Test completions endpoint with valid messages and API key returns expected response."""
    messages = [{"role": "user", "content": "Hello"}]
    data = {"messages": messages}
    response = client.post("/chat/completions", json=data, headers={"X-API-Key": valid_api_key})
    assert response.status_code == 200


    # Mock the OpenAI passthrough for more comprehensive test
    with patch('main.openai_passthrough', return_value={"id": "1", "choices": [{"text": "Hi there!"}]}) as mock:
        response = client.post("/chat/completions", json=data, headers={"X-API-Key": valid_api_key})
        assert response.json() == {"id": "1", "choices": [{"text": "Hi there!"}]}


def test_api_key_validation_invalid(mock_redis):
    """Ensure API key validation for invalid keys raises an error."""
    messages = [{"role": "user", "content": "Hello"}]
    data = {"messages": messages}
    response = client.post("/chat/completions", json=data, headers={"X-API-Key": "invalid_api_key"})
    assert response.status_code == 401


def test_generic_exception_handler(mock_redis):
    """Test if generic exception handler returns proper error response."""
    with patch('main.generic_exception_handler', side_effect=Exception("Test error")):
        response = client.get("/non-existent-endpoint")
        assert response.status_code == 500
        assert "Internal server error" in response.json()["error"]