import teos10

tempK, presPa, sal = 273.15, 101_325, 0.035_165_04

Gpure = teos10.gibbs.purewater(tempK, presPa)
Gsalt = teos10.gibbs.saline(tempK, presPa, sal)

# Gpure should be zero
# Gpure is almost correct
