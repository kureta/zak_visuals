import queue

from torch import multiprocessing as mp


class Edge:
    def __init__(self, *args, **kwargs):
        self.q = mp.Queue(*args, **kwargs)

    def put(self, item, block=False, timeout=0.02):
        try:
            self.q.put(item, block=block, timeout=timeout)
            return True
        except queue.Full:
            return False

    def get(self, *args, **kwargs):
        return self.q.get(*args, **kwargs)


class BaseNode(mp.Process):
    def __init__(self, pause_event: mp.Event = None):
        super().__init__()
        self.exit = mp.Event()
        self.pause = pause_event

    def run(self):
        self.setup()
        while not self.exit.is_set():
            if self.pause is not None:
                self.pause.wait()
            self.task()

    def setup(self):
        pass

    def task(self):
        raise NotImplementedError

    def join(self, **kwargs) -> None:
        self.exit.set()
        super(BaseNode, self).join()
