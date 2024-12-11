import numpy as np

import teos10

# Check values from IAPWS08 Tables 8-10
temperatures = np.array([273.15, 353, 273.15])
pressures = np.array([101_325, 101_325, 1e8]) / teos10.constants.dbar_to_Pa
salinitys = (
    np.array([0.035_165_04, 0.1, 0.035_165_04]) / teos10.constants.salinity_to_salt
)
check_table8 = {
    "gibbs": [0.101_342_742e3, -0.101_342_742e3, 0],  # J kg-1
    "dG_dS": [0, 0.639_974_067e5, 0.639_974_067e5],  # J kg-1
    "dG_dT": [0.147_643_376, -0.147_643_376, 0],  # J kg-1 K-1
    "dG_dp": [0.100_015_694e-2, -0.274_957_224e-4, 0.972_661_217e-3],  # m3 kg-1
    "d2G_dSdp": [0, -0.759_615_412e3, -0.759_615_412e-3],  # m3 kg-1
    "d2G_dT2": [-0.154_473_542e2, 0.852_861_151, -0.145_944_931e2],  # J kg-1 K-2
    "d2G_dTdp": [-0.677_700_318e-7, 0.119_286_787e-6, 0.515_167_556e-7],  # m3 kg-1 K-1
    "d2G_dp2": [
        -0.508_928_895e-12,
        0.581_535_172e-13,
        -0.450_775_377e-12,
    ],  # m3 kg-1 Pa-1
    "enthalpy": [0.610_139_535e2, -0.610_139_535e2, 0.0],  # J kg-1
    "helmholtz_energy": [0.183_99e-2, -0.985_567_377e2, -0.985_548_978e2],  # J kg-1
    "internal_energy": [-0.403_269_484e2, -0.582_279_494e2, -0.985_548_978e2],  # J kg-1
    "entropy": [-0.147_643_376, 0.147_643_376, 0.0],  # J kg-1 K-1
    "density": [0.999_843_086e3, np.nan, 0.102_810_720e4],  # kg m-3
    "heat_capacity": [0.421_944_481e4, -0.232_959_023e3, 0.398_648_579e4],  # J kg-1 K-1
    "sound_speed": [0.140_238_253e4, np.nan, 0.144_900_246e4],  # m s-1
    "chemical_potential_water": [
        0.101_342_742e3,
        -0.235_181_411e4,
        -0.225_047_137e4,
    ],  # J kg-1
}
check_table9 = {
    "gibbs": [-0.446_114_969e5, 0.150_871_740e5, -0.295_243_229e5],  # J kg-1
    "dG_dS": [0.0, 0.251_957_276e6, 0.251_957_276e6],  # J kg-1
    "dG_dT": [-0.107_375_993e4, 0.156_230_907e3, -0.917_529_024e3],  # J kg-1 K-1
    "dG_dp": [0.102_892_956e-2, -0.579_227_286e-4, 0.971_006_828e-3],  # m3 kg-1
    "d2G_dSdp": [0.0, -0.305_957_802e-3, -0.305_957_802e-3],  # m3 kg-1
    "d2G_dT2": [-0.118_885_000e2, 0.127_922_649e1, -0.106_092_735e2],  # J kg-1 K-2
    "d2G_dTdp": [0.659_051_552e-6, 0.803_061_596e-6, 0.146_211_315e-5],  # m3 kg-1 K-1
    "d2G_dp2": [
        -0.474_672_819e-12,
        0.213_086_154e-12,
        -0.261_586_665e-12,
    ],  # m3 kg-1 Pa-1
    "enthalpy": [0.334_425_759e6, -0.400_623_363e5, 0.294_363_423e6],  # J kg-1
    "helmholtz_energy": [-0.447_157_532e5, 0.150_930_430e5, -0.296_227_102e5],  # J kg-1
    "internal_energy": [0.334_321_503e6, -0.400_564_673e5, 0.294_265_035e6],  # J kg-1
    "entropy": [0.107_375_993e4, -0.156_230_907e3, 0.917_529_024e3],  # J kg-1 K-1
    "density": [0.971_883_832e3, np.nan, 0.102_985_888e4],  # kg m-3
    "heat_capacity": [0.419_664_050e4, -0.451_566_952e3, 0.374_507_355e4],  # J kg-1 K-1
    "sound_speed": [0.155_446_297e4, np.nan, 0.396_127_835e4],  # m s-1
    "chemical_potential_water": [
        -0.446_114_969e5,
        -0.101_085_536e5,
        -0.547_200_505e5,
    ],  # J kg-1
}
check_table10 = {
    "gibbs": [0.977_303_862e5, -0.260_093_051e4, 0.951_294_557e5],  # J kg-1
    "dG_dS": [0.0, -0.545_861_581e4, -0.545_861_581e4],  # J kg-1
    "dG_dT": [0.851_466_502e1, 0.754_045_685e1, 0.160_551_219e2],  # J kg-1 K-1
    "dG_dp": [0.956_683_329e-3, -0.229_123_842e-4, 0.933_770_945e-3],  # m3 kg-1
    "d2G_dSdp": [0.0, -0.640_757_619e-3, -0.640_757_619e-3],  # m3 kg-1
    "d2G_dT2": [0.142_969_873e2, 0.488_076_974, -0.138_089_104e2],  # J kg-1 K-2
    "d2G_dTdp": [0.199_079_571e-6, 0.466_284_412e-7, 0.245_708_012e-6],  # m3 kg-1 K-1
    "d2G_dp2": [
        0.371_530_889e-12,
        0.357_345_736e-13,
        -0.335_796_316e-12,
    ],  # m3 kg-1 Pa-1
    "enthalpy": [0.954_046_055e5, -0.466_060_630e4, 0.907_439_992e5],  # J kg-1
    "helmholtz_energy": [0.206_205_330e4, -0.309_692_089e3, 0.175_236_121e4],  # J kg-1
    "internal_energy": [0.263_727_446e3, -0.236_936_788e4, -0.263_309_532e4],  # J kg-1
    "entropy": [0.851_466_502e1, -0.754_045_685e1, -0.160_551_219e2],  # J kg-1 K-1
    "density": [0.104_527_796e4, np.nan, 0.107_092_645e4],  # kg m-3
    "heat_capacity": [0.390_522_209e4, -0.133_318_225e3, 0.377_190_387e4],  # J kg-1 K-1
    "sound_speed": [0.157_542_240e4, np.nan, 0.162_198_998e4],  # m s-1
    "chemical_potential_water": [
        0.977_303_862e5,
        -0.240_897_806e4,
        0.953_214_082e5,
    ],  # J kg-1
}


def factor_sigfig(x):
    return 10.0 ** np.ceil(np.log10(np.abs(x)))


def sigfig(x, sf):
    """Return `x` to `sf` significant figures."""
    factor = 10.0 ** np.ceil(np.log10(np.abs(x)))
    return factor * np.round(x / factor, decimals=sf)


def formatter(values):
    return ("{:.8e} " * len(values)).format(*sigfig(values, 9))


# def test_gibbs_water():
#     """Compare pure water Gibbs energy with check values from IAPWS09."""
#     test_values = []
#     for t, p in zip(temperature_water, pressure_water):
#         test_values.append(teos10.gibbs.water(t, p))
#     assert formatter(check_water["gibbs"]) == formatter(test_values)


# def test_dG_dT_water():
#     """Compare temperature derivative of pure water Gibbs energy with check values from
#     IAPWS09.
#     """
#     test_values = []
#     for t, p in zip(temperature_water, pressure_water):
#         test_values.append(teos10.properties.dG_dT(teos10.gibbs.water)(t, p))
#     assert formatter(check_water["dG_dT"]) == formatter(test_values)


# def test_dG_dp_water():
#     """Compare pressure derivative of pure water Gibbs energy with check values from
#     IAPWS09.
#     """
#     test_values = []
#     for t, p in zip(temperature_water, pressure_water):
#         test_values.append(teos10.properties.dG_dp(teos10.gibbs.water)(t, p))
#     assert formatter(check_water["dG_dp"]) == formatter(test_values)


# def test_d2G_dT2_water():
#     """Compare second temperature derivative of pure water Gibbs energy with check
#     values from IAPWS09.
#     """
#     test_values = []
#     for t, p in zip(temperature_water, pressure_water):
#         test_values.append(teos10.properties.d2G_dT2(teos10.gibbs.water)(t, p))
#     assert formatter(check_water["d2G_dT2"]) == formatter(test_values)


# def test_d2G_dTdp_water():
#     """Compare temperature-pressure derivative of pure water Gibbs energy with check
#     values from IAPWS09.
#     """
#     test_values = []
#     for t, p in zip(temperature_water, pressure_water):
#         test_values.append(teos10.properties.d2G_dTdp(teos10.gibbs.water)(t, p))
#     assert formatter(check_water["d2G_dTdp"]) == formatter(test_values)


# def test_d2G_dp2_water():
#     """Compare temperature-pressure derivative of pure water Gibbs energy with check
#     values from IAPWS09.
#     """
#     test_values = []
#     for t, p in zip(temperature_water, pressure_water):
#         test_values.append(teos10.properties.d2G_dp2(teos10.gibbs.water)(t, p))
#     assert formatter(check_water["d2G_dp2"]) == formatter(test_values)


# def test_enthalpy_water():
#     """Compare pure water enthalpy with check values from IAPWS09."""
#     test_values = []
#     for t, p in zip(temperature_water, pressure_water):
#         test_values.append(teos10.properties.enthalpy(t, p, gfunc=teos10.gibbs.water))
#     assert formatter(check_water["enthalpy"]) == formatter(test_values)


# def test_helmholtz_energy_water():
#     """Compare pure water Helmholtz energy with check values from IAPWS09."""
#     test_values = []
#     for t, p in zip(temperature_water, pressure_water):
#         test_values.append(
#             teos10.properties.helmholtz_energy(t, p, gfunc=teos10.gibbs.water)
#         )
#     assert formatter(check_water["helmholtz_energy"]) == formatter(test_values)


# def test_internal_energy_water():
#     """Compare pure water internal energy with check values from IAPWS09."""
#     test_values = []
#     for t, p in zip(temperature_water, pressure_water):
#         test_values.append(
#             teos10.properties.internal_energy(t, p, gfunc=teos10.gibbs.water)
#         )
#     assert formatter(check_water["internal_energy"]) == formatter(test_values)


# def test_entropy_water():
#     """Compare pure water entropy with check values from IAPWS09."""
#     test_values = []
#     for t, p in zip(temperature_water, pressure_water):
#         test_values.append(teos10.properties.entropy(t, p, gfunc=teos10.gibbs.water))
#     assert formatter(check_water["entropy"]) == formatter(test_values)


# def test_density_water():
#     """Compare pure water density with check values from IAPWS09."""
#     test_values = []
#     for t, p in zip(temperature_water, pressure_water):
#         test_values.append(teos10.properties.density(t, p, gfunc=teos10.gibbs.water))
#     assert formatter(check_water["density"]) == formatter(test_values)


# def test_heat_capacity_water():
#     """Compare pure water heat capacity with check values from IAPWS09."""
#     test_values = []
#     for t, p in zip(temperature_water, pressure_water):
#         test_values.append(
#             teos10.properties.heat_capacity(t, p, gfunc=teos10.gibbs.water)
#         )
#     assert formatter(check_water["heat_capacity"]) == formatter(test_values)


# def test_sound_speed_water():
#     """Compare pure water sound speed with check values from IAPWS09."""
#     test_values = []
#     for t, p in zip(temperature_water, pressure_water):
#         test_values.append(
#             teos10.properties.sound_speed(t, p, gfunc=teos10.gibbs.water)
#         )
#     assert formatter(check_water["sound_speed"]) == formatter(test_values)