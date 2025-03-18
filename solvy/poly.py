import numpy as np
import scipy.spatial

class Polynomial:
    def __init__(self, exps, shape, coeffs = None, mse = None, error = None, mean_error = None, storage = None):
        self.nvars = exps.shape[0]
        assert exps.reshape((-1,)+shape).shape[0] == self.nvars
        self.shape = shape
        self.exps = exps
        self.ncoeffs = self.exps.shape[1]
        self.coeffs = coeffs
        self.mse = mse
        self.error = error
        self.mean_error = mean_error
        if storage is None:
            storage = np.empty([1,*self.exps.shape],dtype=coeffs and coeffs.dtype or np.float64)
        else:
            assert storage.shape[1:] == exps.shape
        self.storage = storage

    @property
    def degrees(self):
        return [d - 1 for d in self.shape]

    @classmethod
    def from_nvars(cls, nvars, degree=2):
        return cls._from_degrees(np.full([nvars], degree+1))

    @classmethod
    def from_degrees(cls, *degrees):
        return cls._from_degrees(np.array(degrees) + 1)

    @classmethod
    def from_data(cls, inputs, outputs, approximate=True, degree=None, overdetermine=False):
        assert inputs.shape[0] == outputs.shape[0]
        nvars = inputs.shape[1]
        if degree is None:
            # solve ncoeffs = datapoints = (degree+1)**nvars
            degree = len(outputs) ** (1./nvars)
            if overdetermine:
                degree = np.ceil(degree)
            elif approximate:
                degree /= 2
            degree = int(degree) - 1
        self = cls.from_nvars(nvars, degree=degree)
        self.solve(inputs, outputs)
        return self

    def solve(self, inputs, outputs):
        assert inputs.shape[0] == outputs.shape[0]
        assert inputs.shape[1] == self.nvars
        nrows = len(inputs)
        if self.storage.shape[0] < nrows:
            self.storage = np.empty([nrows,self.nvars,self.ncoeffs],dtype=self.storage.dtype)
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
            self.error = 0
            self.mean_error = 0
        else:
            # this could batch many polynomials at once by simply passing the batch to pinv
            # would involve making a batched polynomial implementation
            coeffs = [
                    np.linalg.pinv(rows,rtol=0) @ outputs,
                    (outputs @ np.linalg.pinv(rows.T,rtol=0)).T,
            ]
            coeffs.append((coeffs[0]+coeffs[1]) / 2)
            coeffs = np.stack(coeffs)
            estimates = []
            for coeff in coeffs:
                self.coeffs = coeff
                estimates.append(self(inputs))
            estimates = np.stack(estimates)
            diffs = estimates - outputs
            mse = (diffs * diffs).sum(axis=-1)
            while True:
                sorted_idcs = np.argsort(mse)
                best_idx = sorted_idcs[0]
                best_coeffs = coeffs[best_idx]
                new_coeffs = coeffs[sorted_idcs[:-1]].mean(axis = 0)
                if (new_coeffs == coeffs).all(axis = 1).any(axis = 0):
                    break
                new_idx = sorted_idcs[-1]
                self.coeffs = new_coeffs
                new_estimates = self(inputs)
                new_diffs = new_estimates - outputs
                if (new_diffs == diffs).all(axis = 1).any(axis = 0):
                    break
                coeffs[new_idx] = new_coeffs
                estimates[new_idx] = new_estimates
                diffs[new_idx] = new_diffs
                new_mse = (new_diffs * new_diffs).sum(axis=-1)
                mse[new_idx] = new_mse
            diffs = diffs[best_idx]
            self.coeffs = coeffs[best_idx]

            # estimate the volume integral of the error
            if self.nvars == 1:
                # could use storage
                inps = np.sort(inputs[:,0])
                bounds = (inps[1:] + inps[:-1]) / 2
                output_weights = np.concatenate([bounds - inps[:-1], inps[-1:] - bounds[-1:]])
            else:
                # this approach iterates the input, there may be a better way
                input_voronoi = scipy.spatial.Voronoi(inputs) # this constructs bounding regions with each bounding ridge halfway between nearest points
                                                              # but at the bounds the ridges go to infinity, represented by a vertex index of -1
                voronoi_edge_ridges = np.array([pts for pts, verts in input_voronoi.ridge_dict.items() if -1 in verts])
                voronoi_vertices = input_voronoi.vertices
                output_weights = []
                for pt_idx in range(inputs.shape[0]):
                    # this could likely almost all be parallelized using e.g. 2D index masks

                    region = input_voronoi.regions[input_voronoi.point_region[pt_idx]]
                    region = np.array(region) # could use storage
                    vertices = voronoi_vertices[region[region != -1]]
                    if len(vertices) < len(region):
                        # off-the-edge, add bounds
                        # this could use review/debugging to verify correctness .. for example i seem to be unnecessarily adding some redundant points
                        edge_ptidx_pairs = voronoi_edge_ridges[np.logical_or(*(voronoi_edge_ridges == pt_idx).T)]
                        edge_vertices = inputs[edge_ptidx_pairs,:].mean(axis=-1)
                        vertices = np.concatenate([vertices,edge_vertices,inputs[pt_idx:pt_idx+1]],axis=0)
                    # calculate the volume of the region as the weight
                    output_weights.append(scipy.spatial.ConvexHull(vertices).volume)
                output_weights = np.stack(output_weights)
            # output_weights now hopefully has the unique units^n volume associated with each input point
            volume_diffs = diffs * output_weights
            total_volume = output_weights.sum()
            self.mse = float((diffs*volume_diffs).sum() / total_volume)
            error = np.abs(volume_diffs).sum()
            self.mean_error = float(error / total_volume)
            self.error = float(error)

    def __call__(self, values):
        assert (len(values.shape) > 0 and values.shape[-1] == self.nvars and len(values.shape) <= 2) or (self.nvars == 1 and len(values.shape) <= 1)
        if len(values.shape) == 0:
            # scalar passed to 1-var polynomial
            nrows = 1
            out_shape = []
            values = values.reshape([1,1])
        elif len(values.shape) == 1:
            if values.shape[0] == self.nvars:
                # vector passed to n-var polynomial
                nrows = 1
                out_shape = []
                values = values.reshape([1,self.nvars])
            else:
                # many scalars passed to 1-var polynomial
                nrows = values.shape[0]
                out_shape = [nrows]
                values = values.reshape([nrows,1])
        else:
            # many vectors passed to n-var polynomial
            nrows = values.shape[0]
            out_shape = [nrows]
        if self.storage.shape[0] < nrows:
            self.storage = np.empty([nrows,self.nvars,self.ncoeffs],dtype=self.storage.dtype)
        storage = self.storage[:nrows]
        np.pow(values[...,None], self.exps, out=storage)
        row = storage[:,0]
        storage.prod(axis=-2,out=row)
        row *= self.coeffs
        return row.sum(axis=-1).reshape(out_shape)

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

    def copy(self):
        return type(self)(self.exps, self.shape, self.coeffs.copy(), mse=self.mse, error=self.error, mean_error=self.mean_error, storage=self.storage)

    def __eq__(self, other):
        assert (self.exps == other.exps).all()
        assert self.shape == other.shape
        return self.coeffs == other.coeffs

    def set(self, other):
        self.exps = other.exps
        self.shape = other.shape
        if self.coeffs.shape == other.coeffs.shape:
            self.coeffs[:] = other.coeffs
        else:
            self.coeffs = other.coeffs.copy()
        self.mse = other.mse
        self.error = other.error
        self.storage = other.storage

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
                if val*val < 1e-12:
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
        if first:
            out += '0'
        return out

    def __repr__(self):
        return f'Polynomial({repr(self.exps)}, {repr(self.shape)}, coeffs={repr(self.coeffs)}, mse={self.mse}, mean_error={self.mean_error})'

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
