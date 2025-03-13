# a very simple function data cache

import bisect

import numpy as np

SLICE_ALL = [slice(None)]

def ndarray_resize(old_ndarray, new_shape):
    # there may be a numpy function for this
    # could also further generalize into an nd-list w insert/exponential capacity
    old_shape = old_ndarray.shape
    #old_length = len(old_ndarray.flat)
    #new_ndarray = old_ndarray
    #new_ndarray.resize(new_shape, refcheck=False)
    #old_ndarray = ndarray.flat[:old_length].reshape(old_shape, copy=False)
    new_ndarray = np.empty(new_shape, dtype=old_ndarray.dtype)
    copy_slice = tuple([
        slice(min(old_shape[idx], new_shape[idx]))
        for idx in range(min(len(old_shape), len(new_shape)))
    ])
    new_ndarray[copy_slice] = old_ndarray[copy_slice]
    return new_ndarray

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
            self.inputs = inputs[...,None].copy()
            self._inputs_range = np.arange(self.ninputs)
            outputs = self.f(inputs)
            self.noutputs = len(outputs)
            self.data = outputs.reshape([1] * self.ninputs + [self.noutputs]).copy()
            return outputs
        else:
            # find indices
            idcs = np.empty(self.ninputs,dtype=int)
            data_shape = list(self.data.shape)
            for idx1 in range(self.ninputs):
                idx_vals = self.inputs[idx1]
                input_val = inputs[idx1]
                idx2 = bisect.bisect_left(idx_vals[:data_shape[idx1]], input_val)
                idcs[idx1] = idx2
                if idx2 >= len(idx_vals) or idx_vals[idx2] != input_val:
                    # insert new data!
                    idx2_tail = idx2 + 1
                    # resize
                    old_idx_len = data_shape[idx1]
                    new_idx_len = old_idx_len + 1
                    data_shape[idx1] = new_idx_len
                    # resize inputs and slide up
                    if self.inputs.shape[-1] < new_idx_len:
                        inputs_shape = list(self.inputs.shape)
                        inputs_shape[-1] = new_idx_len * 2
                        self.inputs = ndarray_resize(self.inputs, inputs_shape)
                        idx_vals = self.inputs[idx1]
                    idx_vals[idx2_tail:new_idx_len] = idx_vals[idx2:old_idx_len]
                    idx_vals[idx2] = input_val
                    # resize data and slide up
                    self.data = ndarray_resize(self.data, data_shape)
                    slices = SLICE_ALL * self.ninputs
                    slices[idx1] = slice(idx2,-1); slice_move_src = tuple(slices)
                    slices[idx1] = slice(idx2_tail,None); slice_move_dst = tuple(slices)
                    self.data[slice_move_dst] = self.data[slice_move_src] # nd-move?
                    # place hyperplane of new data values
                    slices[idx1] = slice(idx2,idx2_tail); slice_insert = tuple(slices)
                    #iter_vals = self.data[slice_insert]#.reshape([-1, self.noutputs], copy=False)
                    iter_shape = data_shape[:-1]
                    iter_shape[idx1] = 1
                    #assert iter_shape == list(iter_vals.shape[:-1])
                    iter_indices = np.indices(iter_shape) ## no longer making assumption: # presently making the assumption that np.indices presents indices in the same order as .reshape([-1,...])
                    #iter_vals = self.data[slice_insert]#[tuple(iter_indices)].reshape([-1, self.noutputs],copy=False)
                    iter_indices = iter_indices.T.reshape(-1, self.ninputs)
                    iter_indices[:,idx1] = idx2
                    #assert iter_vals.shape[0] == iter_indices.shape[0]
                    n_new_vals = iter_indices.shape[0]
                    for idx in range(n_new_vals):
                        idx_inputs = self.inputs[self._inputs_range, iter_indices[idx]]
                        idx_outputs = self.f(idx_inputs)
                        #iter_vals[idx] = idx_outputs
                        #iter_vals[iter_indices[idx], self._inputs_range] = idx_outputs
                        #self.data[iter_indices[idx], self._inputs_range] = idx_outputs
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
