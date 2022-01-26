import threading


def open():
    print('-' * 10)


barrier = threading.Barrier(3, open)


class Customer(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def run(self):
        while True:
            print('{} 等'.format(self.name))
            try:
                barrier.wait(1)
            except threading.BrokenBarrierError:
                pass


if __name__ == '__main__':
    ts = [Customer() for t in range(3)]
    [t.start() for t in ts]
