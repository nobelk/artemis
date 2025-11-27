from abc import ABC, abstractmethod

class Memory(ABC):
    @abstractmethod
    def get(self):
        pass

    @abstractmethod
    def put(self):
        pass
