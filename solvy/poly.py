import numpy as np

class Polynomial:
    def __init__(self, exps, shape, coeffs = None, mse = None, storage = None):
        self.nvars = exps.shape[0]
        assert exps.reshape((-1,)+shape).shape[0] == self.nvars
        self.shape = shape
        self.exps = exps
        self.ncoeffs = self.exps.shape[1]
        self.coeffs = coeffs
        self.mse = mse
        if storage is None:
            storage = np.empty([1,*self.exps.shape])
        else:
            assert storage.shape[1:] == exps.shape
        self.storage = storage

    @classmethod
    def from_nvars(cls, nvars, degree=2):
        return cls._from_degrees(np.full([nvars], degree+1))

    @classmethod
    def from_degrees(cls, *degrees):
        return cls._from_degrees(np.array(degrees) + 1)

    @classmethod
    def from_data(cls, inputs, outputs, degree=None, overdetermine=False):
        assert inputs.shape[0] == outputs.shape[0]
        nvars = inputs.shape[1]
        if degree is None:
            # solve ncoeffs = datapoints = (degree+1)**nvars
            degree = len(outputs) ** (1./nvars) - 1
            if overdetermine:
                degree = np.ceil(degree)
            degree = int(degree)
        self = cls.from_nvars(nvars, degree=degree)
        self.solve(inputs, outputs)
        return self

    def solve(self, inputs, outputs):
        assert inputs.shape[0] == outputs.shape[0]
        assert inputs.shape[1] == self.nvars
        nrows = len(inputs)
        if self.storage.shape[0] < nrows:
            self.storage = np.empty([nrows,self.nvars,self.ncoeffs])
            storage = self.storage
        else:
            storage = self.storage[:nrows]
        storage[:] = inputs[...,None]
        storage **= self.exps
        rows = storage[:,0]
        storage.prod(axis=-2,out=rows)
        if nrows == self.ncoeffs:
            self.coeffs = np.linalg.solve(rows, outputs)
            self.mse = 0
        else:
            self.coeffs, self.mse, rank, singulars = np.linalg.lstsq(rows, outputs)

    def __call__(self, values):
        storage = self.storage[0]
        np.pow(values[...,None], self.exps, out=storage)
        row = storage[0]
        storage.prod(axis=-2,out=row)
        row *= self.coeffs
        return row.sum()

    def differentiate(self, var_idx):
        # multiply by exponent into coefficient that decrements exponent
        coeffs_nd = self.coeffs.reshape(self.shape)
        exps_nd = self.exps[var_idx].reshape(self.shape)
        slices = [slice(None)] * self.nvars
        slices[var_idx] = slice(1,None); slices_in = tuple(slices)
        slices[var_idx] = slice(0,-1); slices_out = tuple(slices)
        slices[var_idx] = slice(-1,None); slices_zeroed = tuple(slices)
        np.multiply(
            coeffs_nd[slices_in],
            exps_nd[slices_in],
            out=coeffs_nd[slices_out]
        )
        coeffs_nd[slices_zeroed] = 0

    def clone(self):
        return type(self)(self.exps, self.shape, self.coeffs, mse=self.mse, storage=self.storage)

    def __str__(self):
        varnames = 'xyzwabcdefghijklmnopqrstuv'
        coeffnames = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        out = ''
        first = True
        for cidx in range(self.ncoeffs-1,-1,-1):
            if self.coeffs is None:
                if first:
                    first = False
                else:
                    out += ' + '
                out += coeffnames[self.ncoeffs-1-cidx]
            else:
                val = self.coeffs[cidx]
                val = np.float32(val)
                if val*val < 1e-6:
                    continue
                if first:
                    first = False
                else:
                    if val < 0:
                        val = -val
                        out += ' - '
                    else:
                        out += ' + '
                out += str(val)
            for vidx in range(self.nvars):
                exp = self.exps[vidx][cidx]
                if exp:
                    out += varnames[vidx]
                    if exp != 1:
                        out += '^' + str(exp)
        return out

    @classmethod
    def _from_degrees(cls, degrees):
        nvars = len(degrees)
        exps = np.indices(degrees)
        return cls(exps.reshape([nvars,-1]), exps.shape[1:])
    

if __name__ == '__main__':
    for inputs, outputs in [
        [ np.array([
            [0,0],
            [0,1],
            [1,0],
            [1,1],
            [.5,.5]
        ]), np.array([
            0,0,0,0,
            1,
        ]) ],
    ]:
        poly = Polynomial.from_degrees(2,2)
        print('f(x,y) =', str(poly))
        poly.solve(inputs, outputs)
        print('f(x,y) =', str(poly))
        for idx in range(len(inputs)):
            assert np.abs(poly(inputs[idx]) - outputs[idx]) < 1e-6
        poly.differentiate(1)
        print('df(x,y)/dy =', str(poly))
        poly.differentiate(0)
        print('ddf(x,y)/dydx =', str(poly))

