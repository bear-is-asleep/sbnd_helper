from parent import CAF
from mcprim import MCPRIM
import unittest
from pandas import DataFrame
import numpy as np


class TestMCPRIM(unittest.TestCase):
    def setUp(self):
        # Create a simple MCPRIM instance for use in the tests
        data = {'col1': [1, 2], 'col2': [3, 4]}
        self.mcprim = MCPRIM(data)

    def test_getitem_returns_mcprim(self):
        # Test that slicing the MCPRIM instance returns an MCPRIM instance
        result = self.mcprim['col1']
        self.assertIsInstance(result, MCPRIM)

    def test_setitem_returns_mcprim(self):
        # Test that setting an item on the MCPRIM instance maintains the MCPRIM type
        self.mcprim['col3'] = [5, 6]
        self.assertIsInstance(self.mcprim, MCPRIM)

    def test_dataframe_methods_return_mcprim(self):
        # Test that DataFrame methods return an MCPRIM instance
        result = self.mcprim.copy()
        self.assertIsInstance(result, MCPRIM)

        result = self.mcprim.drop('col1', axis=1)
        self.assertIsInstance(result, MCPRIM)

        # Continue this with other DataFrame methods as needed

unittest.main()
