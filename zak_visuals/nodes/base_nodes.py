import queue

from torch import multiprocessing as mp


class Edge:
    def __init__(self, maxsize=1):
        self.q = mp.Queue(maxsize=maxsize)

    def write(self, value):
        try:
            self.q.put(value)
        except queue.Full:
            return

    def read(self):
        try:
            value = self.q.get()
        except queue.Empty:
            value = None
        return value

    def flush(self):
        item = 1
        while item is not None:
            try:
                item = self.q.get(block=False)
            except queue.Empty:
                break

    def cleanup_output(self):
        self.flush()

    def cleanup_input(self):
        self.q.put(None)
        self.flush()
        self.q.close()
        self.q.join_thread()


class BaseNode:
    def __init__(self):
        self.processor = mp.Process(target=self.process)
        self.exit = mp.Event()

    def start(self):
        self.processor.start()

    def stop(self):
        print(f'Exiting {self.__class__.__name__}.')
        self.exit.set()
        self.cleanup()
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

    def cleanup(self):
        raise NotImplementedError
