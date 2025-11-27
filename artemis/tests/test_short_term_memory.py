import sys
from pathlib import Path

import pytest

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.short_term_memory import ShortTermMemory
from src.agent.memory import Memory


class TestShortTermMemory:
    """Test suite for ShortTermMemory class"""

    def test_initialization(self):
        """Test that ShortTermMemory initializes correctly"""
        memory = ShortTermMemory()
        assert hasattr(memory, '_messages')
        assert isinstance(memory._messages, list)
        assert len(memory._messages) == 0

    def test_inheritance(self):
        """Test that ShortTermMemory inherits from Memory"""
        memory = ShortTermMemory()
        assert isinstance(memory, Memory)

    def test_get_empty_messages(self):
        """Test get() returns empty list when no messages added"""
        memory = ShortTermMemory()
        result = memory.get()
        assert result == []
        assert isinstance(result, list)

    def test_get_with_messages(self):
        """Test get() returns messages after they are added"""
        memory = ShortTermMemory()
        # Directly add messages to _messages
        test_messages = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there!'}
        ]
        memory._messages = test_messages

        result = memory.get()
        assert result == test_messages
        assert len(result) == 2

    def test_get_returns_same_reference(self):
        """Test that get() returns a reference to the internal messages list"""
        memory = ShortTermMemory()
        test_message = {'role': 'user', 'content': 'Test'}
        memory._messages.append(test_message)

        result = memory.get()
        # Verify it's the same reference
        assert result is memory._messages

    def test_messages_mutability(self):
        """Test that messages can be modified through the reference"""
        memory = ShortTermMemory()
        messages = memory.get()

        # Add a message through the reference
        messages.append({'role': 'user', 'content': 'New message'})

        # Verify it's reflected in the memory
        assert len(memory._messages) == 1
        assert memory._messages[0]['content'] == 'New message'

    def test_put_method_exists(self):
        """Test that put() method exists"""
        memory = ShortTermMemory()
        assert hasattr(memory, 'put')
        assert callable(memory.put)