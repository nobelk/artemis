import sys
from pathlib import Path

import pytest
from unittest.mock import Mock, MagicMock, patch

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.summarization_based_memory import SummarizationBasedMemory
from src.agent.memory import Memory


class TestSummarizationBasedMemory:
    """Test suite for SummarizationBasedMemory class"""

    @pytest.fixture
    def mock_bot(self):
        """Create a mock Assistant bot"""
        bot = Mock()
        bot.run = MagicMock()
        return bot

    def test_initialization_with_defaults(self, mock_bot):
        """Test that SummarizationBasedMemory initializes with default parameters"""
        memory = SummarizationBasedMemory(bot=mock_bot)

        assert hasattr(memory, '_messages')
        assert isinstance(memory._messages, list)
        assert len(memory._messages) == 0
        assert memory._bot == mock_bot
        assert memory._min_length == 3
        assert memory._max_length == 5

    def test_initialization_with_custom_parameters(self, mock_bot):
        """Test initialization with custom min and max length"""
        memory = SummarizationBasedMemory(
            bot=mock_bot,
            min_length=5,
            max_length=10
        )

        assert memory._min_length == 5
        assert memory._max_length == 10
        assert memory._bot == mock_bot

    def test_inheritance(self, mock_bot):
        """Test that SummarizationBasedMemory inherits from Memory"""
        memory = SummarizationBasedMemory(bot=mock_bot)
        assert isinstance(memory, Memory)

    def test_get_with_empty_messages(self, mock_bot):
        """Test get() with no messages"""
        mock_bot.run.return_value = iter([
            {'role': 'assistant', 'content': 'No conversation to summarize.'}
        ])

        memory = SummarizationBasedMemory(bot=mock_bot)
        result = memory.get()

        # Should still create a summary even with empty messages
        assert isinstance(result, str)
        mock_bot.run.assert_called_once()

    def test_get_with_user_assistant_messages(self, mock_bot):
        """Test get() creates summary from user and assistant messages"""
        expected_summary = "User greeted. Assistant responded politely."
        mock_bot.run.return_value = iter([
            {'role': 'assistant', 'content': expected_summary}
        ])

        memory = SummarizationBasedMemory(bot=mock_bot, min_length=2, max_length=3)
        memory._messages = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there!'},
            {'role': 'user', 'content': 'How are you?'},
            {'role': 'assistant', 'content': 'I am doing well, thank you!'}
        ]

        result = memory.get()

        assert result == expected_summary

        # Verify the bot was called with correct parameters
        mock_bot.run.assert_called_once()
        call_args = mock_bot.run.call_args
        messages = call_args[1]['messages']

        assert len(messages) == 1
        assert messages[0]['role'] == 'user'
        assert 'Summarize this conversation in 2-3 sentences' in messages[0]['content']
        assert 'user: Hello' in messages[0]['content']
        assert 'assistant: Hi there!' in messages[0]['content']

    def test_get_filters_system_messages(self, mock_bot):
        """Test that get() only includes user and assistant messages, not system"""
        expected_summary = "Brief conversation summary."
        mock_bot.run.return_value = iter([
            {'role': 'assistant', 'content': expected_summary}
        ])

        memory = SummarizationBasedMemory(bot=mock_bot)
        memory._messages = [
            {'role': 'system', 'content': 'You are a helpful assistant'},
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi!'},
            {'role': 'system', 'content': 'Another system message'}
        ]

        result = memory.get()

        # Verify system messages are not included in summary prompt
        call_args = mock_bot.run.call_args
        messages = call_args[1]['messages']
        content = messages[0]['content']

        assert 'system:' not in content
        assert 'user: Hello' in content
        assert 'assistant: Hi!' in content

    def test_get_with_multiple_response_items(self, mock_bot):
        """Test that get() returns the last item from bot response"""
        mock_bot.run.return_value = iter([
            {'role': 'assistant', 'content': 'First response'},
            {'role': 'assistant', 'content': 'Second response'},
            {'role': 'assistant', 'content': 'Final summary'}
        ])

        memory = SummarizationBasedMemory(bot=mock_bot)
        memory._messages = [
            {'role': 'user', 'content': 'Test'}
        ]

        result = memory.get()

        assert result == 'Final summary'

    def test_get_formats_conversation_correctly(self, mock_bot):
        """Test that conversation is formatted correctly for summarization"""
        mock_bot.run.return_value = iter([
            {'role': 'assistant', 'content': 'Summary'}
        ])

        memory = SummarizationBasedMemory(bot=mock_bot, min_length=1, max_length=2)
        memory._messages = [
            {'role': 'user', 'content': 'First message'},
            {'role': 'assistant', 'content': 'First response'}
        ]

        memory.get()

        call_args = mock_bot.run.call_args
        messages = call_args[1]['messages']
        content = messages[0]['content']

        # Check format is "role: content"
        assert content == (
            'Summarize this conversation in 1-2 sentences:\n\n'
            'user: First message\n'
            'assistant: First response'
        )

    def test_put_method_exists(self, mock_bot):
        """Test that put() method exists"""
        memory = SummarizationBasedMemory(bot=mock_bot)
        assert hasattr(memory, 'put')
        assert callable(memory.put)

    def test_get_with_empty_content_messages(self, mock_bot):
        """Test get() handles messages with empty content"""
        mock_bot.run.return_value = iter([
            {'role': 'assistant', 'content': 'Summary of minimal conversation.'}
        ])

        memory = SummarizationBasedMemory(bot=mock_bot)
        memory._messages = [
            {'role': 'user', 'content': ''},
            {'role': 'assistant', 'content': 'Response'}
        ]

        result = memory.get()

        assert isinstance(result, str)
        # Verify it still creates the summary prompt correctly
        call_args = mock_bot.run.call_args
        messages = call_args[1]['messages']
        assert 'user: \n' in messages[0]['content']

    @patch('src.agent.summarization_based_memory.Assistant')
    def test_integration_with_real_assistant_class(self, mock_assistant_class, mock_bot):
        """Test that the class can work with the actual Assistant import"""
        memory = SummarizationBasedMemory(bot=mock_bot)
        assert memory._bot == mock_bot
