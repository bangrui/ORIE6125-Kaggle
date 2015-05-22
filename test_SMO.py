from SMO import *
import numpy as np
import unittest

class TestSMOOnePoint(unittest.TestCase):
    def setUp(self):
        X = np.array([[1,1]])
        Y = np.array([1])
        self.SMOobj = SMO(X, Y, 0.1, 0.001, 0.001)
        self.SMOobj.SMO_main()

    def test_weight_correctness(self):
        self.assertEqual(self.SMOobj.get_weight(), np.array([0.0, 0.0]))

    def test_threshold_correctness(self):
        self.assertEqual(self.SMOobj.get_threshold(), 0.0)


class TestSMOTwoPoints(unittest.TestCase):
    def setUp(self):
        X = np.array([[-1,-1], [1,1]])
        Y = np.array([-1, 1])
        self.SMOobj = SMO(X, Y, 0.1, 0.001, 0.001)

    def test_weight_correctness(self):
        self.assertTrue(self.SMOobj.get_weight()[0]/self.SMOobj.get_weight()[1] >= 1 - 1e-5 
                and self.SMOobj.get_weight()[0]/self.SMOobj.get_weight()[1] <= 1 + 1e-5)

    def test_threshold_correctness(self):
        self.assertTrue(self.SMOobj.get_threshold() <= 1e-5 and self.SMOobj.get_threshold() >= -1e-5) 


