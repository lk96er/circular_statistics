# Circular Statistics Tests in Python

This repository contains Python implementations of three important tests for circular (directional) statistics:

1. Kuiper's Test of Uniformity
2. Watson's Two-Sample Test of Homogeneity
3. Rayleigh Test of Uniformity


## Original Implementation

The original implementations of these tests were written in R by:

**Author:** Claudio Agostinelli  
**E-mail:** claudio@unive.it

## Python Implementation

This Python port was created by Lucas Kühl.

## Installation

To use these tests, you need Python 3.6 or later. The implementation depends on NumPy and SciPy.

1. Clone this repository:
   ```
   git clone https://github.com/lk96er/circular-statistics.git
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Here are some examples of how to use the tests:

```python
import numpy as np
from circular_statistics import KuiperTest, WatsonTwoTest, RayleighTest

# Generate some random circular data
x = np.random.uniform(0, 2*np.pi, 100)
y = np.random.uniform(0, 2*np.pi, 100)

# Kuiper's Test
kuiper_test = KuiperTest(x, alpha=0.05)
kuiper_test.run_test()
kuiper_test.print_results()

# Watson's Two-Sample Test
watson_test = WatsonTwoTest(x, y, alpha=0.05)
watson_test.run_test()
watson_test.print_results()

# Rayleigh Test
rayleigh_test = RayleighTest(x)
rayleigh_test.run_test()
rayleigh_test.print_results()

# Rayleigh Test with specified mean direction
rayleigh_test_mu = RayleighTest(x, mu=np.pi)
rayleigh_test_mu.run_test()
rayleigh_test_mu.print_results()
```

## Tests Description

1. **Kuiper's Test**: Tests the null hypothesis that a sample of circular data is drawn from a uniform distribution.

2. **Watson's Two-Sample Test**: Tests the null hypothesis that two samples of circular data come from the same distribution.

3. **Rayleigh Test**: Tests the null hypothesis that a sample of circular data is uniformly distributed around the circle. It can also test for a specified mean direction.

## Contributing

Contributions to improve the code or add new circular statistics tests are welcome. Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the GNU General Public License version 2 (GPL-2) - see the LICENSE file for details.

## References

Jammalamadaka, S. R. and SenGupta, A. (2001). Topics in Circular Statistics. World Scientific, Singapore.


## Acknowledgments

Special thanks to Claudio Agostinelli for the original R implementation of these tests.
