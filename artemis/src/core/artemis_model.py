import random

class ArtemisModel:
    def __init__(self):
        random.seed(7919)

    def get_random_int(self)->int:
        return random.randint(0, 100)

    def get_random_float(self)->float:
        return random.random()
