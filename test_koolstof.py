from autograd import numpy as np
import teos10

tempK = np.array([273.15, 353, 273.15])
presPa = np.array([101_325, 101_325, 1e8])
sal = np.array([0.035_165_04, 0.1, 0.035_165_04])

Gpure = teos10.gibbs.purewater(tempK, presPa)
Gsalt = teos10.gibbs.saline(tempK, presPa, sal)
Gsea = teos10.gibbs.seawater(tempK, presPa, sal)

# Gpure should be zero
# Gpure is almost correct

density = teos10.properties.density(tempK, presPa, sal)

print(density)
