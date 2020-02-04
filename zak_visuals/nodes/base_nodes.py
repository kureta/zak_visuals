from torch import multiprocessing as mp


class BaseNode(mp.Process):
    def run(self):
        self.setup()
        while True:
            self.task()

    def setup(self):
        pass

    def task(self):
        raise NotImplementedError
