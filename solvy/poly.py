import numpy as np

class Degree:
    def __init__(self, *degrees):
        self.nvars = len(degrees)
        self.varexps = np.indices([d+1 for d in degrees]).reshape([self.nvars,-1])
        self.ncoeffs = len(self.varexps[0])
        self.storage = np.empty([0,self.nvars,self.ncoeffs])
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
        storage **= self.varexps[None]
        rows = storage[:,0]
        storage.prod(axis=-2,out=rows)
        if nrows == self.ncoeffs:
            coeffs = np.linalg.solve(rows, outputs)
        else:
            coeffs, residuals, rank, singulars = np.linalg.lstsq(rows, outputs)
        return Polynomial(coeffs, self.varexps)

class Polynomial:
    def __init__(self, coeffs, exps):
        assert coeffs.shape[0] == exps.shape[1]
        self.nvars, self.ncoeffs = exps.shape
        self.coeffs = coeffs
        self.exps = exps
        self.storage = np.empty(exps.shape, dtype=coeffs.dtype)
    def __call__(self, values):
        storage = self.storage
        storage[:] = values[...,None]
        storage **= self.exps
        row = storage[0]
        storage.prod(axis=-2,out=row)
        row *= self.coeffs
        return row.sum()
    def differentiate(self, var_idx):
        raise NotImplementedError('implement polynomial differentiation')
    

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
        poly = Degree(2,2).solve(inputs, outputs)
        for idx in range(len(inputs)):
            assert np.abs(poly(inputs[idx]) - outputs[idx]) < 1e-6

