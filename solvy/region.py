# data for a region can be generated from a mapping of one group of variables to another

# when approximating a region, we could treat any group of variables as input


import numpy as np

# the idea is to slightly/gently abstract the approximator
# and let the user set the number of points to sample (8 for double-sampled 3rd degree polynomials)
# and the approximator adjusts itself to half the datapoint count (.from_data(approximate=True))
# and then some fitting algorithm detects the highest mse and splits along that. (we could differentiate the polynomial with regard to each input to figure out the most impacting input to split on).
    #


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
        minmaxes = np.array(minmaxes_in)
        mins, maxes = minmaxes.T
        diffs = maxes - mins

        idcs_nd = np.indices(nsamps_in)
        shape = idcs_nd.shape[1:]
        idcs = idcs_nd.reshape([nin,-1])
        nsamps_total = idcs.shape[1]

        ins = (idcs.T * diffs / (nsamps_in - 1) + mins)
        ins_nd = ins.reshape(shape + (nin,))

        self.minmaxes_in = minmaxes
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

# for now Mapping is separate from DataRegion to help clarify
# that they both have inputs and outputs, but these may differ.
# DataRegion uses inputs to fill the outputs of func in its data.
# Mapping treats values as inputs for approximating others as outputs.
class Mapping:
    def __init__(self, region, input_idcs, output_idcs, Approximation):
        self.region = region
        self.inputs = input_idcs
        self.outputs = output_idcs
        self.maps = [
            Approximation.from_data(self.region.data[:,self.inputs], self.region.data[:,output])
            for output in output_idcs
        ]
        self.worst_output = np.argmax(self.mses)
    @property
    def mses(self):
        return np.array([map.mse for map in self.maps])
    @property
    def worst_mse(self):
        return self.maps[self.worst_output].mse
    def split_worst(self):
        worst_val = self.outputs[self.worst_output]
        print('The worst value is #', worst_val, 'with an mse of', self.worst_mse)
        worst_map = self.maps[self.worst_output]
        print(worst_val, '=', worst_map)
        # polynomial class could provide extrema
        # instead of estimating with abs and max here
        max_vals = np.abs(self.region.minmaxes_in).max(axis=-1)
        max_mag = -np.inf
        diff = worst_map.copy()
        for idx in range(len(mapping.inputs)):
            diff.set(worst_map)
            diff.differentiate(idx)
            print('d',worst_val,'/ d', mapping.inputs[idx], '=', diff)
            np.abs(diff.coeffs, out=diff.coeffs)
            mag = diff(max_vals)
            if mag > max_mag:
                worst_input = idx
                max_mag = mag
        print('The value most impactful to the error is #', mapping.inputs[worst_input], 'with a maximum derivative of', max_mag)

        # try every split point
        #for split_idx in range(0, 
    def __str__(self):
        return '\n'.join([str(map) for map in self.maps])

if __name__ == '__main__':
    def root_and_square(inputs):
        return np.concatenate([np.sqrt(inputs), inputs**2])

    data = DataRegion(root_and_square, np.array([[1,4],[1,16]]), 4, nsamps_in=7)
    #print('value then root then square')
    #print(data.data_nd)

    import poly
    mapping = Mapping(data, np.array([0,1]), np.array([2,3,4,5]), poly.Polynomial)
    print(mapping.worst_mse)
    mapping.split_worst()

