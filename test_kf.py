from kf import KF
import unittest
import numpy as np

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

    def test_after_calling_predict_mean_and_cov_are_of_right_shape(self):
        x = 0.2
        v = 2.3

        kf = KF(initial_x = x, initial_v = v, accel_variance= 1.2)

        kf.predict(dt = 0.1)
        self.assertEqual(kf.cov.shape, (2,2))
        self.assertEqual(kf.mean.shape, (2,))
        

    def test_after_calling_predict_increases_state_uncertainty(self):
        x = 0.2
        v = 2.3
        temp = 0
        kf = KF(initial_x = x, initial_v = v, accel_variance= 1.2)
        for  i in range(10):
            #l'entropie de la matrice de covariance est exprimé par son determinant, ici je m'attend à obtenir un determinant qui croit.
            #l'incertitude devant être de plus en plus grande au cours du temps.
            det_before = np.linalg.det(kf.cov)
            kf.predict(dt=0.1)
            det_after = np.linalg.det(kf.cov)
            
            self.assertGreater(det_after, det_before)
            #print(det_before, det_after)

    def test_update_ok(self):
        x = 0.2
        v = 2.3

        kf = KF(initial_x= x, initial_v = v, accel_variance = 1.2)
        kf.update(meas_value = 0.1, meas_variance = 0.1)

    def test_calling_update_decreases_state_uncertainty(self):
        x = 0.2
        v = 2.3

        kf = KF(initial_x = x, initial_v=v,accel_variance = 1.2)
        kf.update(meas_value=0.1, meas_variance= 0.1)

        for i in range(10):

            det_before = np.linalg.det(kf.cov)
            kf.update(meas_value = 0.1, meas_variance= 0.01)
            det_after = np.linalg.det(kf.cov)


            self.assertLess(det_after, det_before)
            print(det_after, det_before)