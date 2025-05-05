import pytest
import json
from main import generate_cache_key

def test_generate_cache_key_with_valid_messages():
    """
    Test that generate_cache_key produces the expected cache key for a typical input.
    """
    messages = [
        {'role': 'user', 'content': 'Hello, how are you?'},
        {'role': 'assistant', 'content': 'I am fine, thank you!'}
    ]
    expected_key = 'cache:5d1791b29c79204d41c9bc474e62f682c99a06bc7c083582c62d650dad30e3a8'
    assert generate_cache_key(messages) == expected_key


def test_generate_cache_key_with_empty_list():
    """
    Test that generate_cache_key produces the expected cache key for an empty input.
    """
    messages = []
    expected_key = 'cache:0'*64
    assert generate_cache_key(messages) == expected_key


def test_generate_cache_key_order_independence():
    """
    Test that the cache key is the same regardless of the order of messages.
    """
    messages_1 = [
        {'role': 'user', 'content': 'Hello'},
        {'role': 'assistant', 'content': 'Hi!'}
    ]
    messages_2 = [
        {'role': 'assistant', 'content': 'Hi!'},
        {'role': 'user', 'content': 'Hello'}
    ]
    assert generate_cache_key(messages_1) == generate_cache_key(messages_2)