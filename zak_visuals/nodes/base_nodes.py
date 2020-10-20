import time

from torch import multiprocessing as mp


class BaseNode(mp.Process):
    def __init__(self, pause_event: mp.Event = None):
        super().__init__()
        self.exit = mp.Event()
        self.pause = pause_event

    def run(self):
        self.setup()
        zzz = 1 / 120
        while not self.exit.is_set():
            t0 = time.time()
            if self.pause:
                self.pause.wait()
            self.task()
            t1 = time.time()
            wait = zzz - (t1 - t0)
            time.sleep(max(wait, 0))

    def setup(self):
        pass

    def task(self):
        raise NotImplementedError

    def join(self, **kwargs) -> None:
        self.exit.set()
        super(BaseNode, self).join()
