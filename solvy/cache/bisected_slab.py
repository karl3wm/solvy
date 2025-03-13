# a very simple function data cache

import bisect

import numpy as np

from .ndlist import NDList

class BisectedSlab:
    def __init__(self, n2n_f):
        self.f = n2n_f
        self.ninputs = None
        self.inputs = None
        self._data = None
        self.noutputs = None

    def bounds(self):
        return np.stack([self.inputs[:,0], self.inputs[self._inputs_range, self._data.shape[:-1]-1]])

    def data(self, bounds=None):
        startends = (np.array(bounds) if bounds else self.bounds()).T
        idcs = self._inputs2idcs(startends)
        densities = idcs[:,1] - idcs[:,0]
        idcs = np.indices(densities)
        idcs = (idcs.reshape(self.ninputs,-1).T + idcs[:,0])

        # hum inputs outputs (now just 'data')
        # also some food interest
        # thing karl likes to um let calm, he's unsure how
        outputs = self._data[*idcs.T]
        return np.concatenate([
            self.inputs[self._inputs_range, idcs],
            self._data[*idcs.T],
        ],axis=1)

#    def outputs(self, bounds=None):
#        startends = (np.array(bounds) if bounds else self.bounds()).T
#        idcs = self._inputs2idcs(startends)
#        return self._data(*[
#            slice(start, end)
#            for start, end in idcs
#        ]).view(-1, self.noutputs)
#
#    def inputs(self, bounds, sparse=False):
#        startends = np.array(bounds).T
#        idcs = self._inputs2idcs(startends)
#
#        if sparse:
#            return [
#                self.inputs[idx,slice(*idcs[idx])]
#                for idx in self.ninputs
#            ]
#        else:
#            # lets start by collecting all the coordinates of the region.
#            densities = idcs[:,1] - idcs[:,0]
#            all_idcs = np.indices(densities)
#            shape = all_idcs.shape[1:]
#            all_idcs = (all_idcs.reshape(self.ninputs,-1).T + idcs[:,0])
#
#            inputs = self.inputs[all_idcs]
#            return inputs.reshape(*shape, self.ninputs)

    def get_densities(self, bounds):
        startends = np.array(bounds).T
        idcs = self._inputs2idcs(startends)
        return idcs[:,1] - idcs[:,0]

    def ensure_density(self, bounds, density, endpoint=False):
        bounds = np.array(bounds)
        if endpoint:
            step = (bounds[1] - bounds[0]) / density
            inputslist = np.linspace(bounds[0], bounds[1] + step, density + 2)
        else:
            inputslist = np.linspace(bounds[0], bounds[1], density + 1)

        if self.inputs is None:
            # first data
            for inputs in inputslist[:-1]:
                self[inputs]
        else:
            idcs = self._inputs2idcs(inputslist.T)
            # insert data for any idcs that are equal
            missing_idcs = idcs[:,:-1] == idcs[:,1:]

            # this could likely be optimized
            for idx1 in range(self.ninputs):
                for idx2 in range(len(inputslist) - 1):
                    if missing_idcs[idx1,idx2]:
                        inputs = self.inputs[:,0].copy()
                        # new input is calculated by linearly interpolating the entire range
                        inputs[idx1] = (bounds[:,idx1] * [density-idx2,idx2]).sum() / density
                        self[inputs] # generate new data

    def _inputs2idcs(self, inputs):
        return np.stack([
            np.searchsorted(self.inputs[idx][:self._data.shape[idx]], inputs[idx])
            for idx in range(self.ninputs)
        ])

    def __call__(self, *inputs):
        return self[np.array(inputs)]

    def __getitem__(self, inputs):
        if self.inputs is None:
            # first coord
            self.ninputs = len(inputs)
            self.inputs = NDList(inputs[...,None])
            self._inputs_range = np.arange(self.ninputs)
            outputs = self.f(inputs)
            self.noutputs = len(outputs)
            self._data = NDList(outputs.reshape([1] * self.ninputs + [self.noutputs]))
            return outputs
        else:
            # find indices
            idcs = self._inputs2idcs(inputs)
            old_data_shape = self._data.shape
            if (old_data_shape[:-1] == self.inputs.shape[1]).any():
                # i added a note to NDList._insert_empty about a simple way to generalize to inserting an element of ragged data
                self.inputs.resize([self.ninputs, self.inputs.shape[1]+1])
            new_input_mask = np.logical_or(
                idcs >= old_data_shape[:-1] ,#or
                self.inputs[self._inputs_range, idcs] != inputs
            )
            #if new_input_mask.any():
            #    self._data._insert_empty(idcs, new_input_mask)
            #    new_data_shape = self._data.shape
            cur_data_shape = self._data.shape
            for idx1 in range(self.ninputs):
                if new_input_mask[idx1]:
                    # insert new data!
                    input_val = inputs[idx1]
                    idx2 = idcs[idx1]

                    # slide up inputs
                    # i added a note to NDList._insert_empty about a simple way to generalize to inserting an element of ragged data
                    idx_vals = self.inputs[idx1]
                    idx_vals[idx2+1:old_data_shape[idx1]+1] = idx_vals[idx2:old_data_shape[idx1]]
                    # place new input
                    idx_vals[idx2] = input_val

                    # resize data and slide up
                    insert_where, insert_expansion = np.zeros([2,self.ninputs+1],dtype=int)
                    insert_where[idx1] = idx2
                    insert_expansion[idx1] = 1
                    self._data._insert_empty(insert_where, insert_expansion)
                    cur_data_shape = self._data.shape

                    # place hyperplane of new data values
                    iter_shape = list(cur_data_shape[:-1])
                    iter_shape[idx1] = 1
                    iter_indices = np.indices(iter_shape)
                    iter_indices = iter_indices.T.reshape(-1, self.ninputs)
                    iter_indices[:,idx1] = idx2
                    n_new_vals = iter_indices.shape[0]
                    for idx in range(n_new_vals):
                        idx_inputs = self.inputs[self._inputs_range, iter_indices[idx]]
                        idx_outputs = self.f(idx_inputs)
                        self._data[tuple(iter_indices[idx])] = idx_outputs

            return self._data[tuple(idcs)]


if __name__ == '__main__':
    sqrtslab2 = BisectedSlab(np.sqrt)
    assert np.allclose(sqrtslab2(4,16), [2,4])
    assert np.allclose(sqrtslab2(4,16), [2,4])
    assert np.allclose(sqrtslab2(9, 4), [3,2])
    assert np.allclose(sqrtslab2(4,16), [2,4])
    assert np.allclose(sqrtslab2(25,9), [5,3])
    assert np.allclose(sqrtslab2(4,16), [2,4])
    sqrtslab2 = BisectedSlab(np.sqrt)
    sqrtslab2.ensure_density([[0,0], [4,4]], 2, endpoint=True)
    assert np.allclose(sqrtslab2.inputs.data[:,:3], [[0,2,4], [0,2,4]])
    sqrtslab2.ensure_density([[0,0], [4,4]], 4, endpoint=True)
    assert np.allclose(sqrtslab2.inputs.data[:,:5], [[0,1,2,3,4], [0,1,2,3,4]])
    sqrtslab2.ensure_density([[8,10], [10,12]], 4, endpoint=True)
    assert np.allclose(sqrtslab2.inputs.data[:,:10], [[0,1,2,3,4,8,8.5,9,9.5,10], [0,1,2,3,4,10,10.5,11,11.5,12]])
    sqrtslab2 = BisectedSlab(np.sqrt)
    sqrtslab2.ensure_density([[0,0], [4,4]], 2, endpoint=False)
    assert np.allclose(sqrtslab2.inputs.data[:,:2], [[0,2], [0,2]])
    sqrtslab2.ensure_density([[0,0], [4,4]], 4, endpoint=False)
    assert np.allclose(sqrtslab2.inputs.data[:,:4], [[0,1,2,3], [0,1,2,3]])
    sqrtslab2.ensure_density([[8,10], [10,12]], 4, endpoint=False)
    assert np.allclose(sqrtslab2.inputs.data[:,:8], [[0,1,2,3,8,8.5,9,9.5], [0,1,2,3,10,10.5,11,11.5]])
