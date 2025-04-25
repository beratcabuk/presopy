
import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_database():
    # Fixture to provide a mock database connection for tests
    mock_db = MagicMock()
    mock_db.connect.return_value = True
    mock_db.execute_query.return_value = ["result1", "result2"]
    return mock_db

@pytest.fixture
def mock_config():
    # Fixture to provide mock configuration
    return {
        "database": {
            "host": "localhost",
            "port": 5432,
            "username": "test_user",
            "password": "test_password",
            "database": "test_db"
        },
        "api": {
            "port": 8000,
            "debug": True,
            "timeout": 30
        }
    }
