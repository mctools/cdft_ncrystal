import numpy as np

# FunctionXY is a helper class for HDRFT calculation.
# The constructor parameters are defined by the equations followed
# The function is y=f(x)

# asymExponent describes the symetric between the positive and nagtive
# part of the function: f(-x) = exp(x*asymExponent) * f(x).
# so when asymExponent==0., f(-x) = f(x)

# distortFact is the parameter that defines the tilting distortion when
# performing the convolution for the HDRFT process
# distort() function scales the function by np.exp(x*distortFact), while
# the recover() function removes the distortion. Notice that, the asymExponent
# factor is automaticlly taken into account when performing these two funcitons.

# For example, the classic scattering function asymExponent==0. and distortFact
# can be the detailed balance factor.

class Curve():
    def __init__(self, x, y):
        if np.any (np.diff (x) <= 0.):
            raise RuntimeError(f'x of the vector is not sorted')

        if x.size != y.size:
            raise RuntimeError(f'x and y are different in size')

        self.__x = np.copy(x)
        self.__y = np.copy(y)

    #make x and y read only from outside
    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    def scaleY(self, factor):
        self.__y = self.__y*factor

    def normalise(self):
        self.__y *= 1./np.trapz(self.__y, self.__x)

    def crop(self, minX, maxX):
        idx1 = None
        idx2 = None
        if self.__x[0] < minX:
            idx1=np.searchsorted(self.__x, minX)
        if self.__x[-1] > maxX:
            idx2=np.searchsorted(self.__x, maxX, 'right')

        self.__x = self.__x[idx1:idx2]
        self.__y = self.__y[idx1:idx2]


class LinSpacedCurve(Curve):
    def __init__(self, x, y):
        super().__init__(x, y)
        if not self.isLinSpacedCruve():
            raise RuntimeError(f'x is not evently spaced')

    def isLinSpacedCruve(self):
        if hasattr(self, 'isLinSpaced'):
            return self.isLinSpaced
        else: #notice this is a slow process, so we do it only once for an object
            self.isLinSpaced = self.checkLinSpaced()
            return self.isLinSpaced

    def checkLinSpaced(self):
        diff = np.diff(self._Curve__x)
        diff *= 1./self.getDeltaX()
        np.testing.assert_almost_equal(diff, np.ones(diff.size), decimal = 8, err_msg='not evently spaced')
        return True

    def getDeltaX(self):
        return (self.x[-1]-self.x[0])/(self.x.size-1)

    def accumulate(self, aLinSpacedCurve):
        if not aLinSpacedCurve.isLinSpacedCruve():
            raise RuntimeError(f'aLinSpacedCurve is not evently spaced')

        s1 = self.getDeltaX()
        s2 = aLinSpacedCurve.getDeltaX()
        np.testing.assert_almost_equal(s1/s2, 1., decimal = 14, err_msg='different x spacing')

        offset= int(round((aLinSpacedCurve.x[0]-self.x[0])/s1))
        # ---------------        self.x
        #      ----------------  input.x
        if offset >= 0 :
            if aLinSpacedCurve.y.size <= self.x.size-offset:
                self._Curve__y[offset:(offset+aLinSpacedCurve.y.size)] += aLinSpacedCurve.y
            else: #extend the back of self.x and self.y
                np.testing.assert_almost_equal(self.x[offset]/aLinSpacedCurve.x[0], 1., decimal = 13)
                extrapoints = aLinSpacedCurve.y.size-(self.x.size-offset)
                self._Curve__y = np.concatenate((self._Curve__y, np.zeros(extrapoints)))
                self._Curve__y[offset:(offset+aLinSpacedCurve.y.size)] += aLinSpacedCurve.y
                self._Curve__x = np.concatenate((self._Curve__x, aLinSpacedCurve.x[-extrapoints:]))
        #      ---------------  self.x
        # ----------------      input.x
        else:
            # extend the front of x and y
            self._Curve__x = np.concatenate((aLinSpacedCurve.x[:-offset], self._Curve__x))
            self._Curve__y = np.concatenate((np.zeros(-offset), self._Curve__y))

            # if the back of self.x needs to be extaned as well
            extrapointsAtTheBack = aLinSpacedCurve.x.size-self.x.size
            if extrapointsAtTheBack > 0:
                self._Curve__y = np.concatenate((self._Curve__y, np.zeros(extrapointsAtTheBack)))
                self._Curve__x = aLinSpacedCurve.x

            if aLinSpacedCurve.y.dtype != self._Curve__y.dtype:
                self._Curve__y = self._Curve__y.astype(aLinSpacedCurve.y.dtype)
            self._Curve__y[:(aLinSpacedCurve.y.size)] += aLinSpacedCurve.y

        s1 = self.getDeltaX()
        s2 = aLinSpacedCurve.getDeltaX()
        self.checkLinSpaced()
        np.testing.assert_almost_equal(s1/s2, 1., decimal = 14, err_msg='different x spacing 2')

        self.checkLinSpaced()


class FunctionXY(LinSpacedCurve):
    def __init__(self, x, y, distortFact=0., asymExponent=0., autodistort=False):
        super().__init__(x, y)
        self.distortFact = distortFact #the distortion factor
        self.asymExponent = asymExponent
        z = np.searchsorted(self.x, 0)
        np.testing.assert_almost_equal(self.x[z], 0., decimal = 13, err_msg='x axis should contain zero')

    def distort(self):
        if self.distortFact!=0.:
            self.asymExponent += 2*self.distortFact
            self._Curve__y *= np.exp(self._Curve__x*self.distortFact)

    def restort(self):
        if self.distortFact!=0.:
            self.asymExponent += -2*self.distortFact
            self._Curve__y *= np.exp(-self._Curve__x*self.distortFact)

    def getZ(self):
        z = np.searchsorted(self.x, 0)
        np.testing.assert_almost_equal(self.x[z], 0., decimal = 13, err_msg='x axis should contain zero')
        return z


    def flipNeg2Pos(self):
        z = self.getZ()
        #zero of X belongs to both positive and negtive parts
        posiSize = self.y.size-z
        negSize = z+1

        #if positive part contains too many data points to restort
        if posiSize>negSize:
            self.crop(self.x[0], -self.x[0]*(1.+1e-14)) #fixme: assuming x is not super dense
        else:
            extrapoints = negSize-posiSize
            self._Curve__y = np.concatenate((self._Curve__y, np.zeros(extrapoints)))
            self._Curve__x = np.concatenate((self._Curve__x, -self.x[:extrapoints]))
        if self.x.size != self.y.size:
            raise RuntimeError('x and y have different size')
        newpositive = self.y[:negSize]*np.exp(-self.x[:negSize]*self.asymExponent)
        self._Curve__y[z:]=np.flip(newpositive)
