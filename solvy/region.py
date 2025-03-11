# data for a region can be generated from a mapping of one group of variables to another

# when approximating a region, we could treat any group of variables as input


import numpy as np

class DataRegion:
    def __init__(self, many_to_many_func, minmaxes_in, nout, nsamps_in = 3):
        assert nsamps_in >= 2 # to include upper and lower bound
        func = many_to_many_func
        nin = len(minmaxes_in)
        if type(nsamps_in) is int:
            nsamps_in = np.array([nsamps_in] * nin)
        else:
            assert nsamps_in.shape == (nin,)
        nval = nin + nout
        bounds = np.array(minmaxes_in)
        mins, maxes = bounds.T
        diffs = maxes - mins

        idcs_nd = np.indices(nsamps_in)
        shape = idcs_nd.shape[1:]
        idcs = idcs_nd.reshape([nin,-1])
        nsamps_total = idcs.shape[1]

        ins = (idcs.T * diffs / (nsamps_in - 1) + mins)
        ins_nd = ins.reshape(shape + (nin,))

        #self.bounds_in = bounds
        self.lower_in = mins
        self.upper_in = maxes
        self.func = func
        self.shape = shape

        self.data = np.stack([
            np.concatenate([v, self.func(v)])
            for v in ins
        ])

        # this shape is now such that it can be indexed one axis per input,
        # and the associated input value is strictly increasing along that
        # axis in the data.
        # the final dimension contains the inputs and outputs in order.
        self.data_nd = self.data.reshape(shape + (nval,))


if __name__ == '__main__':
    def root_and_square(inputs):
        return np.concatenate([np.sqrt(inputs), inputs**2])

    data = DataRegion(root_and_square, np.array([[1,4]]), 2, nsamps_in=7)
    print('value then root then square')
    print(data.data_nd)

    import poly
    root_poly = poly.Polynomial.from_data(data.data[:,:1],data.data[:,1],degree=4)
    sqr_poly = poly.Polynomial.from_data(data.data[:,:1],data.data[:,2],degree=4)
    print('root =', root_poly)
    print('root.mse = ', root_poly.mse)
    print('square =', sqr_poly)
    print('square.mse = ', sqr_poly.mse)
