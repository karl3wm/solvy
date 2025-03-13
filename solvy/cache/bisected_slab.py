# a very simple function data cache

import bisect

import numpy as np

from .ndlist import NDList

class BisectedSlab:
    def __init__(self, n2n_f):
        self.f = n2n_f
        self.ninputs = None
        self.inputs = None
        self.data = None
        self.noutputs = None
    def bounds(self):
        return np.stack([self.inputs[0], self.inputs[self.data.shape[:-1]]])
    def __call__(self, inputs):
        return self[inputs]
    def __getitem__(self, inputs):
        if self.inputs is None:
            # first coord
            self.ninputs = len(inputs)
            self.inputs = NDList(inputs[...,None])
            self._inputs_range = np.arange(self.ninputs)
            outputs = self.f(inputs)
            self.noutputs = len(outputs)
            self.data = NDList(outputs.reshape([1] * self.ninputs + [self.noutputs]))
            return outputs
        else:
            # find indices
            idcs = np.empty(self.ninputs,dtype=int)
            for idx1 in range(self.ninputs):
                idx_vals = self.inputs[idx1]
                input_val = inputs[idx1]
                data_shape = self.data.shape
                idx2 = bisect.bisect_left(idx_vals[:data_shape[idx1]], input_val)
                idcs[idx1] = idx2
                if idx2 >= len(idx_vals) or idx_vals[idx2] != input_val:
                    # insert new data!

                    # resize inputs and slide up
                        # - this would be very similar to an nd insertion,
                        #   except that only one axis is slid,
                        #   rather than all of them
                        # - it's also like an insertion into a subarray
                        #   where the outer array storage must be reserved,
                        #   but only the subarray is slid
                       # i added a note to NDList._insert_empty about a simple way to generalize to this
                    self.inputs.resize([
                        self.ninputs,
                        max(self.inputs.shape[1], data_shape[idx1] + 1)
                    ])
                    idx_vals = self.inputs[idx1]
                    idx_vals[idx2+1:data_shape[idx1]+1] = idx_vals[idx2:data_shape[idx1]]
                    idx_vals[idx2] = input_val

                    # resize data and slide up
                    insert_where, insert_expansion = np.zeros([2,self.ninputs+1],dtype=int)
                    insert_where[idx1] = idx2
                    insert_expansion[idx1] = 1
                    self.data._insert_empty(insert_where, insert_expansion)

                    # place hyperplane of new data values
                    iter_shape = list(data_shape[:-1])
                    iter_shape[idx1] = 1
                    iter_indices = np.indices(iter_shape)
                    iter_indices = iter_indices.T.reshape(-1, self.ninputs)
                    iter_indices[:,idx1] = idx2
                    n_new_vals = iter_indices.shape[0]
                    for idx in range(n_new_vals):
                        idx_inputs = self.inputs[self._inputs_range, iter_indices[idx]]
                        idx_outputs = self.f(idx_inputs)
                        self.data[tuple(iter_indices[idx])] = idx_outputs

            return self.data[tuple(idcs)]


if __name__ == '__main__':
    sqrtslab2 = BisectedSlab(np.sqrt)
    assert np.allclose(sqrtslab2(np.array([4,16])), np.array([2,4]))
    assert np.allclose(sqrtslab2(np.array([4,16])), np.array([2,4]))
    assert np.allclose(sqrtslab2(np.array([9,4])), np.array([3,2]))
    assert np.allclose(sqrtslab2(np.array([4,16])), np.array([2,4]))
    assert np.allclose(sqrtslab2(np.array([25,9])), np.array([5,3]))
    assert np.allclose(sqrtslab2(np.array([4,16])), np.array([2,4]))
