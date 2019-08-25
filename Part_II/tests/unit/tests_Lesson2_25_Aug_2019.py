#testcases for Lesson2_25_Aug_2019.py

import pytest
import sys,os
import io
import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../..')

import Lesson2_25_Aug_2019
from fixtures import fixtures as fx

fixtures = []
fixtures = fx.fixtures_gen()
strings = fixtures[1]
ints = fixtures[0]

class TestObject(object):
    def test_numpy_arithmatic_example(self):
        