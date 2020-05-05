from autograd.numpy import sqrt
from autograd import elementwise_grad as egrad
from . import gibbs

# Get differentials
gt = egrad(gibbs.purewater, argnum=0)
gp = egrad(gibbs.purewater, argnum=1)
gtt = egrad(gt, argnum=0)
gtp = egrad(gt, argnum=1)
gpp = egrad(gp, argnum=1)

# Define functions for solution properties
def rho(tempK, presPa):
    """Density in kg/m**3."""
    # Table 3, Eq. (4)
    return 1 / gp(tempK, presPa)


def s(tempK, presPa):
    """Specific entropy in J/(kg*K)."""
    # Table 3, Eq. (5)
    return -gt(tempK, presPa)


def cp(tempK, presPa):
    """Specific isobaric heat capacity in J/(kg*K)."""
    # Table 3, Eq. (6)
    return -tempK * gtt(tempK, presPa)


def h(tempK, presPa):
    """Specific enthalpy in J/kg."""
    # Table 3, Eq. (7)
    return gibbs.purewater(tempK, presPa) + tempK * s(tempK, presPa)


def u(tempK, presPa):
    """Specific internal energy in J/kg."""
    # Table 3, Eq. (8)
    return gibbs.purewater(tempK, presPa) + tempK * s(tempK, presPa) - presPa * gp(tempK, presPa)


def f(tempK, presPa):
    """Specific Helmholtz energy in J/kg."""
    # Table 3, Eq. (9)
    return gibbs.purewater(tempK, presPa) - presPa * gp(tempK, presPa)


def alpha(tempK, presPa):
    """Thermal expansion coefficient in 1/K."""
    # Table 3, Eq. (10)
    return gtp(tempK, presPa) / gp(tempK, presPa)


def bs(tempK, presPa):
    """Isentropic temp.-presPas. coefficient, adiabatic lapse rate in K/Pa."""
    # Table 3, Eq. (11)
    return -gtp(tempK, presPa) / gp(tempK, presPa)


def kt(tempK, presPa):
    """Isothermal compresPasibility in 1/Pa."""
    # Table 3, Eq. (12)
    return -gpp(tempK, presPa) / gp(tempK, presPa)


def ks(tempK, presPa):
    """Isentropic compresPasibility in 1/Pa."""
    # Table 3, Eq. (13)
    return (gtp(tempK, presPa) ** 2 - gtt(tempK, presPa) * gpp(tempK, presPa)) / (
        gp(tempK, presPa) * gtt(tempK, presPa)
    )


def w(tempK, presPa):
    """Speed of sound in m/s."""
    # Table 3, Eq. (14)
    return gp(tempK, presPa) * sqrt(
        gtt(tempK, presPa)
        / (gtp(tempK, presPa) ** 2 - gtt(tempK, presPa) * gpp(tempK, presPa))
    )
