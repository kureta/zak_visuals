import queue

from torch import multiprocessing as mp


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


class InputNode(BaseNode):
    def __init__(self, outgoing: mp.Queue):
        super().__init__()
        self._outgoing = outgoing

    @property
    def outgoing(self):
        raise PermissionError('Not allowed to read from output!')

    @outgoing.setter
    def outgoing(self, value):
        self._outgoing.put(value)

    def run(self):
        raise NotImplementedError


class OutputNode(BaseNode):
    def __init__(self, incoming: mp.Queue):
        super().__init__()
        self._incoming = incoming

    @property
    def incoming(self):
        try:
            value = self._incoming.get(timeout=1)
        except queue.Empty:
            value = None
        return value

    @incoming.setter
    def incoming(self, value):
        raise PermissionError('Not allowed to write to input!')

    def run(self):
        raise NotImplementedError


class ProcessorNode(BaseNode):
    def __init__(self, incoming: mp.Queue, outgoing: mp.Queue):
        super().__init__()
        self._incoming = incoming
        self._outgoing = outgoing

    @property
    def incoming(self):
        try:
            value = self._incoming.get(timeout=1)
        except queue.Empty:
            value = None
        return value

    @incoming.setter
    def incoming(self, value):
        raise PermissionError('Not allowed to write to input!')

    @property
    def outgoing(self):
        raise PermissionError('Not allowed to read from output!')

    @outgoing.setter
    def outgoing(self, value):
        self._outgoing.put(value)

    def run(self):
        raise NotImplementedError
