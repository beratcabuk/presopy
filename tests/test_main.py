import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from main import app, redis_client, settings

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
async def mock_redis():
    redis_client.client = AsyncMock()
    yield
    await redis_client.close()

@pytest.mark.asyncio
async def test_health_check(client, mock_redis):
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

@pytest.mark.asyncio
async def test_cache_stats(client, mock_redis):
    redis_client.client.info.return_value = {"used_memory": 12345, "used_memory_human": "12.3KB", "connected_clients": 5}
    redis_client.client.dbsize = AsyncMock(return_value=50)

    response = client.get("/cache/stats", headers={"X-API-Key": "valid_api_key"})
    assert response.status_code == 200
    assert response.json()["used_memory"] == 12345

@pytest.mark.asyncio
async def test_clear_cache(client, mock_redis):
    redis_client.client.flushdb = AsyncMock()
    response = client.delete("/cache", headers={"X-API-Key": "valid_api_key"})
    assert response.status_code == 200
    assert response.json()["status"] == "success"

@pytest.mark.asyncio
async def test_invalid_api_key(client):
    response = client.get("/cache/stats", headers={"X-API-Key": "invalid_api_key"})
    assert response.status_code == 401

@pytest.mark.asyncio
async def test_rat_limiter(client, mock_redis):
    redis_client.client.zcard = AsyncMock(return_value=settings.rate_limit_requests)
    redis_client.client.zremrangebyscore = AsyncMock()

    response = client.post(
        "/chat/completions",
        headers={"X-API-Key": "valid_api_key", "Authorization": "Bearer token"},
        json={"messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_generate_cache_key():
    messages = [{"role": "user", "content": "Hello"}]
    cache_key = generate_cache_key(messages)
    assert cache_key.startswith("cache:")

@pytest.mark.asyncio
async def test_get_embedding():
    messages = [{"role": "user", "content": "Hello"}]
    auth_header = "Bearer token"
    with patch('httpx.AsyncClient') as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.post = AsyncMock(return_value=AsyncMock(status_code=200, json=AsyncMock(return_value={"data": [{"embedding": [0.1, 0.2, 0.3]}]})))
        embedding = await get_embedding(messages, auth_header)
    assert embedding == [0.1, 0.2, 0.3]

@pytest.mark.asyncio
async def test_openai_passthrough():
    data = {"messages": [{"role": "user", "content": "Hello"}]}
    auth = "Bearer token"
    with patch('httpx.AsyncClient') as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.post = AsyncMock(return_value=AsyncMock(status_code=200, json=AsyncMock(return_value={"choices": [{"message": {"content": "Response Content"}}]})))
        response = await openai_passthrough(auth, data)
    assert response["choices"][0]["message"]["content"] == "Response Content"

@pytest.mark.asyncio
async def test_generic_exception_handler(client):
    response = await client.get("/non-existent-endpoint")
    assert response.status_code == 404