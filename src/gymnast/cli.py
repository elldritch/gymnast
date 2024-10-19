from collections.abc import Callable
import sys
import threading


class InputListener(threading.Thread):
    def __init__(self, callback: Callable[[], None]):
        super().__init__(daemon=True)
        self.callback = callback
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.wait(0.1):
            sys.stdin.readline()
            self.callback()

    def stop(self):
        self.stop_event.set()
