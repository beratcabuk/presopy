import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from main import app, RedisClient

@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture(autouse=True)
async def mock_redis_client(monkeypatch):
    mock_client = AsyncMock(spec=RedisClient)
    monkeypatch.setattr('main.redis_client', mock_client)
    return mock_client


def test_health(client):
    """Verify health check endpoint returns healthy status."""
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'healthy'


def test_cache_stats(client, mock_redis_client):
    """Test retrieval of cache statistics."""
    mock_redis_client.client.info.return_value = {'used_memory': 1024, 'used_memory_human': '1KB', 'connected_clients': 5}
    mock_redis_client.client.dbsize.return_value = 10
    mock_redis_client.get_metrics.return_value = {'total_requests': 0}

    api_key = 'valid_api_key'
    response = client.get('/cache/stats', headers={'X-API-Key': api_key})
    assert response.status_code == 200
    assert response.json()['used_memory'] == 1024


def test_clear_cache(client, mock_redis_client):
    """Test clearing the entire cache."""
    api_key = 'valid_api_key'
    response = client.delete('/cache', headers={'X-API-Key': api_key})
    assert response.status_code == 200
    assert response.json()['status'] == 'success'


def test_rate_limiter(client, mock_redis_client):
    """Test rate limiter logic."""
    mock_redis_client.client.zcard.return_value = 50  # Simulate existing requests
    api_key = 'valid_api_key'
    response = client.post('/chat/completions', headers={'X-API-Key': api_key})
    assert response.status_code == 429  # Rate limit exceeded


@pytest.mark.asyncio
async def test_get_metrics(client, mock_redis_client):
    """Verify metrics retrieval."""
    mock_redis_client.get_metrics.return_value = {'total_requests': 1}
    api_key = 'valid_api_key'
    response = await client.get('/metrics', headers={'X-API-Key': api_key})
    assert response.status_code == 200
    assert response.json() == {'total_requests': 1}


@pytest.mark.asyncio
async def test_chat_completions(client, mock_redis_client):
    """Test chat completions functionality."""
    mock_redis_client.client.set.return_value = True
    mock_redis_client.client.get.return_value = None
    data = {'messages': [{'role': 'user', 'content': 'Hello'}]}  
    api_key = 'valid_api_key'
    response = await client.post('/chat/completions', headers={'X-API-Key': api_key}, json=data)
    assert response.status_code == 200  # Assuming call passes


@pytest.mark.asyncio
async def test_openai_error_handling(client):
    """Test handling of OpenAI API errors."""
    data = {'messages': [{'role': 'user', 'content': 'Error test'}]}  
    api_key = 'valid_api_key'
    with patch('main.get_embedding', side_effect=httpx.HTTPError('OpenAI API error')):
        response = await client.post('/chat/completions', headers={'X-API-Key': api_key}, json=data)
    assert response.status_code == 500  # Should handle correctly
