import time
from collections import deque

class BatchTimer:
    def __init__(self, window_size=100):
        self.times = deque(maxlen=window_size)
        self.last_time = None
    
    def tick(self):
        now = time.time()
        if self.last_time is not None:
            self.times.append(now - self.last_time)
        self.last_time = now

    def tick_end_of_batch(self):
        """Вызывать в конце обработки батча"""
        self.last_time = time.time()  # сброс для следующего замера

    @property
    def avg(self):
        return sum(self.times) / len(self.times) if self.times else 0