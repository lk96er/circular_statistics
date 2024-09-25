# This file is part of Circular Statistics Python.
#
# Circular Statistics Python is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# Circular Statistics Python is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Circular Statistics Python.  If not, see <http://www.gnu.org/licenses/>.

import unittest
import numpy as np
from circular_stats import KuiperTest, WatsonTwoTest, RayleighTest


class TestCircularStatistics(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

        # Generate some test data
        self.uniform_data = np.random.uniform(0, 2 * np.pi, 100)
        self.von_mises_data = np.random.vonmises(np.pi, 5, 100)

    def test_kuiper_test_uniform(self):
        test = KuiperTest(self.uniform_data)
        statistic = test.run_test()
        self.assertIsNotNone(statistic)
        self.assertLess(statistic, 2.001)  # Should not reject uniformity

    def test_kuiper_test_non_uniform(self):
        test = KuiperTest(self.von_mises_data)
        statistic = test.run_test()
        self.assertIsNotNone(statistic)
        self.assertGreater(statistic, 2.001)  # Should reject uniformity

    def test_watson_two_test_same_distribution(self):
        test = WatsonTwoTest(self.uniform_data[:50], self.uniform_data[50:])
        statistic = test.run_test()
        self.assertIsNotNone(statistic)
        self.assertLess(statistic, 0.187)  # Should not reject homogeneity

    def test_watson_two_test_different_distribution(self):
        test = WatsonTwoTest(self.uniform_data, self.von_mises_data)
        statistic = test.run_test()
        self.assertIsNotNone(statistic)
        self.assertGreater(statistic, 0.385)  # Should reject homogeneity

    def test_rayleigh_test_uniform(self):
        test = RayleighTest(self.uniform_data)
        result = test.run_test()
        self.assertIsNotNone(result)
        self.assertGreater(result['p_value'], 0.05)  # Should not reject uniformity

    def test_rayleigh_test_non_uniform(self):
        test = RayleighTest(self.von_mises_data)
        result = test.run_test()
        self.assertIsNotNone(result)
        self.assertLess(result['p_value'], 0.05)  # Should reject uniformity

    def test_rayleigh_test_with_mu(self):
        test = RayleighTest(self.von_mises_data, mu=np.pi)
        result = test.run_test()
        self.assertIsNotNone(result)
        self.assertLess(result['p_value'], 0.05)  # Should reject uniformity

    def test_empty_input(self):
        empty_data = np.array([])

        kuiper_test = KuiperTest(empty_data)
        self.assertIsNone(kuiper_test.run_test())

        watson_test = WatsonTwoTest(empty_data, empty_data)
        self.assertIsNone(watson_test.run_test())

        rayleigh_test = RayleighTest(empty_data)
        self.assertIsNone(rayleigh_test.run_test())

    def test_invalid_alpha(self):
        with self.assertRaises(ValueError):
            KuiperTest(self.uniform_data, alpha=0.2).run_test()


if __name__ == '__main__':
    unittest.main()