from src.agent.memory import Memory

class ShortTermMemory(Memory):
    def __init__(self):
        self._messages = []

    def put(self):
        self._memory.append(self._messages)

    def get(self):
        return self._messages
