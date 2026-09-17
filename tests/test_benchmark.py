import unittest
import numpy as np
from foresight.benchmark import BenchmarkWorld, sensed_state


class BenchmarkTests(unittest.TestCase):
    def test_conditions_preserve_initial_observation(self):
        states=[BenchmarkWorld(7,'crossing',c).reset(seed=7) for c in ('nominal','global_shift','local_patch','switch_recover')]
        for state in states[1:]:np.testing.assert_array_equal(state,states[0])

    def test_shift_has_no_effect_until_scheduled(self):
        a=BenchmarkWorld(3,'open','nominal'); b=BenchmarkWorld(3,'open','switch_recover')
        for _ in range(8):
            sa,*_=a.step(np.array([.3,0]));sb,*_=b.step(np.array([.3,0]))
            np.testing.assert_array_equal(sa,sb)
        sa,*_=a.step(np.array([.3,0]));sb,*_=b.step(np.array([.3,0]))
        self.assertGreater(float(np.linalg.norm(sa[:4]-sb[:4])),0)

    def test_noise_reproducible_and_confined_to_robot(self):
        state=BenchmarkWorld(1,'open','nominal').state
        a=sensed_state(state,'sensor_noise',5,2);b=sensed_state(state,'sensor_noise',5,2)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(a[4:],state[4:])
        self.assertFalse(np.array_equal(a[:4],state[:4]))


if __name__=='__main__':unittest.main()
