from kf import KF
import unittest

class TestKF(unittest.TestCase):
    def test_can_construct_with_x_and_v(self):
        x = 0.2
        v = 2.3

        kf = KF(initial_x = x, initial_v = v, accel_variance= 1.2)
        self.assertAlmostEqual(kf.pos, x)
        self.assertAlmostEqual(kf.vel, v)
    def test_can_predict(self):
        x = 0.2
        v = 2.3

        kf = KF(initial_x = x, initial_v = v, accel_variance= 1.2)

        kf.predict(dt = 0.1)


