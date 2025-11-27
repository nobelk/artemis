from src.agent.memory import Memory
from qwen_agent.agents import Assistant

class SummarizationBasedMemory(Memory):
    def __init__(self, bot:Assistant, min_length=3, max_length=5):
        self._messages = []
        self._bot = bot
        self._min_length = min_length
        self._max_length = max_length

    def put(self):
        self._memory.append(self._messages)

    def get(self):
        """Create a summary of the conversation so far"""
        conversation_text = "\n".join([
            f"{m['role']}: {m['content']}"
            for m in self._messages if m['role'] in ['user', 'assistant']
        ])

        summary_prompt = [{
            'role': 'user',
            'content': f"Summarize this conversation in {self._min_length}-{self._max_length} sentences:\n\n{conversation_text}"
        }]

        summary = list(self._bot.run(messages=summary_prompt))
        return summary[-1]['content']
