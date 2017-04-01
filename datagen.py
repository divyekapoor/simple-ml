from __future__ import print_function
import math
import random
from typing import Generator


class Entry(object):
    def __init__(self, value, label):
        self.value = value
        self.label = label

    def __unicode__(self):
        return '({},{})'.format(self.value, self.label)


def underlying_function(x: float) -> float:
    return 2 * x + 7


def training_datagen() -> Generator[Entry, None, None]:
    for i in range(0, 5000):
        x = float(i) / 1000.
        yield Entry(x, underlying_function(x))


def random_entry_from_training_data() -> Entry:
    x = random.randint(0, 5000) / 1000.
    return Entry(x, underlying_function(x))


def test_datagen() -> Generator[Entry, None, None]:
    for i in range(5000):
        x = random.random() * 5
        yield Entry(x, underlying_function(x))


def categorical_feature(x: float) -> int:
    result = underlying_function(x)
    if result < 0:
        return -1
    elif result > 0:
        return 1
    return 0


def main():
    for i in range(0, 5000):
        x = float(i) / 1000.
        print(x, underlying_function(x))


if __name__ == '__main__':
    main()
