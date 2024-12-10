import numpy as np

import teos10

# Test input values from IAPWS08 Table 8
temperature_salt = np.array([273.15, 353, 273.15])
pressure_salt = np.array([101_325, 101_325, 1e8])
salinity = np.array([0.035_165_04, 0.1, 0.035_165_04])

# Test input values from IAPWS09 Table 6
temperature_water = np.array([273.15, 273.15, 313.15])
pressure_water = np.array([101_325, 1e8, 101_325])


def factor_sigfig(x):
    return 10.0 ** np.ceil(np.log10(np.abs(x)))


def sigfig(x, sf):
    """Return `x` to `sf` significant figures."""
    factor = 10.0 ** np.ceil(np.log10(np.abs(x)))
    return factor * np.round(x / factor, decimals=sf)


def formatter(values):
    return ("{:.8e} " * len(values)).format(*sigfig(values, 9))


def test_gibbs_water():
    """Compare pure water Gibbs energy with check values from IAPWS09."""
    check_values = np.array(
        [
            0.101_342_743 * 10**3,
            0.977_303_868 * 10**5,
            -0.116_198_898 * 10**5,
        ]
    )
    test_values = teos10.gibbs.water(temperature_water, pressure_water)
    assert formatter(check_values) == formatter(test_values)


def test_gibbs_salt():
    """Compare saline component of Gibbs energy with check values from IAPWS08."""
    check_values = np.array(
        [
            -0.101_342_742 * 10**3,
            +0.150_871_740 * 10**5,
            -0.260_093_051 * 10**4,
        ]
    )
    test_values = teos10.gibbs.salt(temperature_salt, pressure_salt, salinity)
    assert formatter(check_values) == formatter(test_values)


# def test_soundSpeed_purewater():
#     """Compare pure water speed of sound with check values from IAPWS09."""
#     check_values = np.array(
#         [0.140_240_099 * 10**4, 0.157_543_089 * 10**4, 0.152_891_242 * 10**4]
#     )
#     assert np.all(
#         sigfig(
#             teos10.properties.soundSpeed(
#                 temperature_water, pressure_water, gibbsfunc=teos10.gibbs.purewater
#             ),
#             9,
#         )
#         - check_values
#         == 0
#     ), "Pure water speed of sound does not match check values from IAPWS09."


# def test_heatCapacity_purewater():
#     """Compare pure water heat capacity with check values from IAPWS09."""
#     check_values = np.array(
#         [0.421_941_153 * 10**4, 0.390_523_030 * 10**4, 0.417_942_416 * 10**4]
#     )
#     assert np.all(
#         sigfig(
#             teos10.properties.heatCapacity(
#                 temperature_water, pressure_water, gibbsfunc=teos10.gibbs.purewater
#             ),
#             9,
#         )
#         - check_values
#         == 0
#     ), "Pure water heat capacity does not match check values from IAPWS09."


# def test_heatCapacity_saline():
#     """Compare saline component of heat capacity with check values from IAPWS08."""
#     check_values = np.array(
#         [-0.232_959_023 * 10**3, -0.451_566_952 * 10**3, -0.133_318_225 * 10**3]
#     )
#     assert np.all(
#         sigfig(
#             teos10.properties.heatCapacity(
#                 temperature_salt, pressure_salt, sal, gibbsfunc=teos10.gibbs.saline
#             ),
#             9,
#         )
#         - check_values
#         == 0
#     ), "Saline component of heat capacity does not match check values from IAPWS08."


# def test_waterChemicalPotential_saline():
#     """Compare saline part of water chemical potential with check values from IAPWS08."""
#     check_values = np.array(
#         [-0.235_181_411 * 10**4, -0.101_085_536 * 10**5, -0.240_897_806 * 10**4]
#     )
#     assert np.all(
#         sigfig(
#             teos10.properties.waterChemicalPotential(
#                 temperature_salt, pressure_salt, sal, gibbsfunc=teos10.gibbs.saline
#             ),
#             9,
#         )
#         - check_values
#         == 0
#     ), (
#         "Saline part of water chemical potential does not match "
#         + "check values from IAPWS08."
#     )
