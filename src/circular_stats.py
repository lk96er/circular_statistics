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

import numpy as np
from scipy import stats


class KuiperTest:
    """
    Kuiper's Test of Uniformity for circular data.

    This class implements Kuiper's test, which is used to test the null hypothesis
    that a sample of circular data is drawn from a uniform distribution.

    Original R implementation:
    Author: Claudio Agostinelli
    E-mail: claudio@unive.it
    URL: https://cran.r-project.org/web/packages/circular/circular.pdf

    Parameters:
    -----------
    x : array-like
        The sample of circular data (in radians).
    alpha : float, optional
        The significance level for the test (default is 0).

    Methods:
    --------
    run_test()
        Performs the Kuiper test on the data.
    print_results(digits=4)
        Prints the results of the test.

    References:
    -----------
    Jammalamadaka, S. R. and SenGupta, A. (2001). Topics in Circular Statistics. World Scientific, Singapore.
    """

    def __init__(self, x, alpha=0):
        self.x = np.array(x)
        self.alpha = alpha
        self.statistic = None

    def run_test(self):
        # Remove NaN values
        self.x = self.x[~np.isnan(self.x)]

        if len(self.x) == 0:
            print("Warning: No observations (at least after removing missing values)")
            return None

        # Convert to radians and ensure values are between 0 and 2π
        self.x = np.mod(self.x, 2 * np.pi)

        self.statistic = self._kuiper_test_rad()
        return self.statistic

    def _kuiper_test_rad(self):
        if self.alpha not in [0, 0.01, 0.025, 0.05, 0.1, 0.15]:
            raise ValueError("'alpha' must be one of the following values: 0, 0.01, 0.025, 0.05, 0.1, 0.15")

        x = np.sort(self.x) / (2 * np.pi)
        n = len(x)
        i = np.arange(1, n + 1)

        D_P = np.max(i / n - x)
        D_M = np.max(x - (i - 1) / n)
        V = (D_P + D_M) * (np.sqrt(n) + 0.155 + 0.24 / np.sqrt(n))

        return V

    def print_results(self, digits=4):
        if self.statistic is None:
            print("Test has not been run yet. Call run_test() first.")
            return

        V = self.statistic
        alpha = self.alpha
        kuiper_crits = {0.15: 1.537, 0.1: 1.62, 0.05: 1.747, 0.025: 1.862, 0.01: 2.001}

        print("\n      Kuiper's Test of Uniformity\n")
        print(f"Test Statistic: {V:.{digits}f}")

        if alpha == 0:
            if V < 1.537:
                print("P-value > 0.15\n")
            elif V < 1.62:
                print("0.10 < P-value < 0.15\n")
            elif V < 1.747:
                print("0.05 < P-value < 0.10\n")
            elif V < 1.862:
                print("0.025 < P-value < 0.05\n")
            elif V < 2.001:
                print("0.01 < P-value < 0.025\n")
            else:
                print("P-value < 0.01\n")
        else:
            critical = kuiper_crits[alpha]
            print(f"Level {alpha} Critical Value: {critical:.4f}")
            if V > critical:
                print("Reject Null Hypothesis\n")
            else:
                print("Do Not Reject Null Hypothesis\n")


# Example usage:
# x = np.random.uniform(0, 2*np.pi, 100)
# test = KuiperTest(x, alpha=0.05)
# test.run_test()
# test.print_results()


class WatsonTwoTest:
    """
    Watson's Two-Sample Test of Homogeneity for circular data.

    This class implements Watson's two-sample test, which is used to test the null
    hypothesis that two samples of circular data come from the same distribution.

    Original R implementation:
    Author: Claudio Agostinelli
    E-mail: claudio@unive.it
    URL: https://cran.r-project.org/web/packages/circular/circular.pdf

    Parameters:
    -----------
    x : array-like
        The first sample of circular data (in radians).
    y : array-like
        The second sample of circular data (in radians).
    alpha : float, optional
        The significance level for the test (default is 0).

    Methods:
    --------
    run_test()
        Performs Watson's two-sample test on the data.
    print_results(digits=4)
        Prints the results of the test.

    References:
    -----------
    Jammalamadaka, S. R. and SenGupta, A. (2001). Topics in Circular Statistics. World Scientific, Singapore.
    """

    def __init__(self, x, y, alpha=0):
        self.x = np.array(x)
        self.y = np.array(y)
        self.alpha = alpha
        self.statistic = None
        self.nx = None
        self.ny = None

    def run_test(self):
        # Remove NaN values
        self.x = self.x[~np.isnan(self.x)]
        self.y = self.y[~np.isnan(self.y)]

        if len(self.x) == 0:
            print("Warning: 'x': No observations (at least after removing missing values)")
            return None

        if len(self.y) == 0:
            print("Warning: 'y': No observations (at least after removing missing values)")
            return None

        # Convert to radians and ensure values are between 0 and 2π
        self.x = np.mod(self.x, 2 * np.pi)
        self.y = np.mod(self.y, 2 * np.pi)

        result = self._watson_two_test_rad()
        self.statistic = result['statistic']
        self.nx = result['nx']
        self.ny = result['ny']
        return self.statistic

    def _watson_two_test_rad(self):
        n1 = len(self.x)
        n2 = len(self.y)
        n = n1 + n2

        x = np.column_stack((np.sort(self.x), np.ones(n1)))
        y = np.column_stack((np.sort(self.y), np.full(n2, 2)))

        xx = np.vstack((x, y))
        rank = np.argsort(xx[:, 0])
        xx = np.column_stack((xx[rank], np.arange(1, n + 1)))

        a = np.cumsum(xx[:, 1] == 1)
        b = np.cumsum(xx[:, 1] == 2)

        d = b / n2 - a / n1
        dbar = np.mean(d)
        u2 = (n1 * n2) / n ** 2 * np.sum((d - dbar) ** 2)

        return {'statistic': u2, 'nx': n1, 'ny': n2}

    def print_results(self, digits=4):
        if self.statistic is None:
            print("Test has not been run yet. Call run_test() first.")
            return

        u2 = self.statistic
        n1 = self.nx
        n2 = self.ny
        alpha = self.alpha
        n = n1 + n2

        print("\n      Watson's Two-Sample Test of Homogeneity\n")
        if n < 18:
            print("Warning: Total Sample Size < 18: Consult tabulated critical values\n")

        crits = {0: 99, 0.001: 0.385, 0.01: 0.268, 0.05: 0.187, 0.1: 0.152}
        print(f"Test Statistic: {u2:.{digits}f}")

        if alpha == 0:
            if u2 > 0.385:
                print("P-value < 0.001\n")
            elif u2 > 0.268:
                print("0.001 < P-value < 0.01\n")
            elif u2 > 0.187:
                print("0.01 < P-value < 0.05\n")
            elif u2 > 0.152:
                print("0.05 < P-value < 0.10\n")
            else:
                print("P-value > 0.10\n")
        else:
            critical = crits.get(alpha, 99)  # Default to 99 if alpha not in crits
            reject = "Reject Null Hypothesis" if u2 > critical else "Do Not Reject Null Hypothesis"
            print(f"Level {alpha} Critical Value: {critical:.{digits}f}")
            print(f"{reject}\n")


# Example usage:
# x = np.random.uniform(0, 2*np.pi, 100)
# y = np.random.uniform(0, 2*np.pi, 100)
# test = WatsonTwoTest(x, y, alpha=0.05)
# test.run_test()
# test.print_results()


class RayleighTest:
    """
    Rayleigh Test of Uniformity for circular data.

    This class implements the Rayleigh test, which is used to test the null hypothesis
    that a sample of circular data is uniformly distributed around the circle.

    Original R implementation:
    Author: Claudio Agostinelli
    E-mail: claudio@unive.it
    URL: https://cran.r-project.org/web/packages/circular/circular.pdf

    Parameters:
    -----------
    x : array-like
        The sample of circular data (in radians).
    mu : float, optional
        The mean direction to test against (in radians). If None, the test is
        performed for the general unimodal alternative.

    Methods:
    --------
    run_test()
        Performs the Rayleigh test on the data.
    print_results(digits=4)
        Prints the results of the test.

    References:
    -----------
    Jammalamadaka, S. R. and SenGupta, A. (2001). Topics in Circular Statistics. World Scientific, Singapore
    """

    def __init__(self, x, mu=None):
        self.x = np.array(x)
        self.mu = mu
        self.statistic = None
        self.p_value = None

    def run_test(self):
        # Remove NaN values
        self.x = self.x[~np.isnan(self.x)]

        if len(self.x) == 0:
            print("Warning: No observations (at least after removing missing values)")
            return None

        # Convert to radians and ensure values are between 0 and 2π
        self.x = np.mod(self.x, 2 * np.pi)

        if self.mu is not None:
            self.mu = np.mod(self.mu, 2 * np.pi)

        result = self._rayleigh_test_rad()
        self.statistic = result['statistic']
        self.p_value = result['p_value']
        return result

    def _rayleigh_test_rad(self):
        n = len(self.x)
        if self.mu is None:
            ss = np.sum(np.sin(self.x))
            cc = np.sum(np.cos(self.x))
            rbar = np.sqrt(ss ** 2 + cc ** 2) / n
            z = n * rbar ** 2
            p_value = np.exp(-z)
            if n < 50:
                temp = 1 + (2 * z - z ** 2) / (4 * n) - (24 * z - 132 * z ** 2 + 76 * z ** 3 - 9 * z ** 4) / (
                        288 * n ** 2)
            else:
                temp = 1
            p_value = np.clip(p_value * temp, 0, 1)
            result = {'statistic': rbar, 'p_value': p_value, 'mu': None}
        else:
            r0_bar = np.sum(np.cos(self.x - self.mu)) / n
            z0 = np.sqrt(2 * n) * r0_bar
            pz = stats.norm.cdf(z0)
            fz = stats.norm.pdf(z0)
            p_value = 1 - pz + fz * ((3 * z0 - z0 ** 3) / (16 * n) +
                                     (15 * z0 + 305 * z0 ** 3 - 125 * z0 ** 5 + 9 * z0 ** 7) / (4608 * n ** 2))
            p_value = np.clip(p_value, 0, 1)
            result = {'statistic': r0_bar, 'p_value': p_value, 'mu': self.mu}
        return result

    def print_results(self, digits=4):
        if self.statistic is None or self.p_value is None:
            print("Test has not been run yet. Call run_test() first.")
            return

        print("\n      Rayleigh Test of Uniformity")
        if self.mu is None:
            print("       General Unimodal Alternative\n")
        else:
            print(f"       Alternative with Specified Mean Direction: {self.mu:.{digits}f}\n")

        print(f"Test Statistic: {self.statistic:.{digits}f}")
        print(f"P-value: {self.p_value:.{digits}f}\n")

# Example usage:
# x = np.random.uniform(0, 2*np.pi, 100)
# test = RayleighTest(x)
# test.run_test()
# test.print_results()

# With specified mean direction:
# test_with_mu = RayleighTest(x, mu=np.pi)
# test_with_mu.run_test()
# test_with_mu.print_results()
