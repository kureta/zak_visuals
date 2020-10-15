from torch import multiprocessing as mp


class BaseNode(mp.Process):
    def __init__(self, pause_event: mp.Event = None):
        super().__init__()
        self.exit = mp.Event()
        self.pause = pause_event

    def run(self):
        self.setup()
        while not self.exit.is_set():
            if self.pause:
                self.pause.wait()
            self.task()

    def setup(self):
        pass

    def task(self):
        raise NotImplementedError

    def join(self, **kwargs) -> None:
        self.exit.set()
        super(BaseNode, self).join()
