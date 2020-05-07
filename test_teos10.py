import numpy as np
import teos10

# Test values from IAPWS08 Table 8
tempK_sal = np.array([273.15, 353, 273.15])
presPa_sal = np.array([101_325, 101_325, 1e8])
sal = np.array([0.035_165_04, 0.1, 0.035_165_04])

# Test values from IAPWS09 Table 6
tempK_pure = np.array([273.15, 273.15, 313.15])
presPa_pure = np.array([101_325, 1e8, 101_325])


def sigfig(x, sf):
    """Return `x` to `sf` significant figures."""
    factor = 10.0 ** np.ceil(np.log10(np.abs(x)))
    return factor * np.around(x / factor, decimals=sf)


def test_gibbs_purewater():
    check_values = np.array(
        [0.101_342_743 * 10 ** 3, 0.977_303_868 * 10 ** 5, -0.116_198_898 * 10 ** 5]
    )
    assert np.all(
        sigfig(teos10.gibbs.purewater(tempK_pure, presPa_pure), 9) - check_values == 0
    ), "Pure water part of Gibbs function does not match check values from IAPWS09."
    

def test_gibbs_saline():
    check_values = np.array(
        [-0.101_342_742 * 10 ** 3, +0.150_871_740 * 10 ** 5, -0.260_093_051 * 10 ** 4]
    )
    assert np.all(
        sigfig(teos10.gibbs.saline(tempK_sal, presPa_sal, sal), 9) - check_values == 0
    ), "Saline part of Gibbs function does not match check values from IAPWS08."

