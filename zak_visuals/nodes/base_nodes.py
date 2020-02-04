from torch import multiprocessing as mp


class BaseNode(mp.Process):
    def __init__(self):
        super().__init__()
        self.exit = mp.Event()

    def run(self):
        self.setup()
        while not self.exit.is_set():
            self.task()

    def setup(self):
        pass

    def task(self):
        raise NotImplementedError

    def join(self, **kwargs) -> None:
        self.exit.set()
        super(BaseNode, self).join()
