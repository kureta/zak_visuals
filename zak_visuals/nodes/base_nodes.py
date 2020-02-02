import queue

from torch import multiprocessing as mp


class Edge:
    def __init__(self):
        self.q = mp.Queue(maxsize=1)

    def write(self, value):
        try:
            self.q.put(value, timeout=1)
        except queue.Full:
            return

    def read(self):
        try:
            value = self.q.get(timeout=1)
        except queue.Empty:
            value = None
        return value


class BaseNode:
    def __init__(self):
        self.processor = mp.Process(target=self.process)
        self.exit = mp.Event()

    def start(self):
        self.processor.start()

    def stop(self):
        print(f'Exiting {self.__class__.__name__}.')
        self.exit.set()
        self.processor.join()
        print(f'{self.__class__.__name__} is kill!')

    def process(self):
        self.setup()
        while not self.exit.is_set():
            self.run()
        self.teardown()

    def setup(self):
        pass

    def run(self):
        raise NotImplementedError

    def teardown(self):
        pass
