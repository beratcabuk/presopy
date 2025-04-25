
import pytest
import redis
from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_redis():
    # Create a mock Redis client
    mock_client = MagicMock(spec=redis.Redis)
    
    # Configure the mock client behavior
    mock_client.get.return_value = b'{"key": "value"}'
    mock_client.set.return_value = True
    mock_client.delete.return_value = 1
    
    return mock_client

def test_redis_set_get(mock_redis):
    # Test Redis set/get operations
    mock_redis.set("test_key", "test_value")
    value = mock_redis.get("test_key")
    
    # Verify
    mock_redis.set.assert_called_once_with("test_key", "test_value")
    mock_redis.get.assert_called_once_with("test_key")
    assert value == b'{"key": "value"}'

def test_redis_cache_expiry(mock_redis):
    # Test cache expiry
    mock_redis.setex("temp_key", 60, "temp_value")
    
    # Verify
    mock_redis.setex.assert_called_once_with("temp_key", 60, "temp_value")
