from src.agent.memory import Memory

class ContextWindowMemory(Memory):
    def __init__(self, max_tokens = 2048):
        self._messages = []
        self.max_tokens = max_tokens

    def put(self):
        self._memory.append(self._messages)

    def get(self):
        system_msgs = [m for m in self._messages if m.get('role') == 'system']
        recent_msgs = self._messages[-(self.max_tokens // 100):]  # Rough estimate
        return system_msgs + recent_msgs
