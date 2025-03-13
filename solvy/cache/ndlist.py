import numpy as np

def ceil_exp_2(data):
    if type(data) in [int, np.int64]:
        return 1 << (data - 1).bit_length()
    else:
        return np.array([(1<<(int(i)-1).bit_length()) for i in data.flat], dtype=data.dtype).reshape(data.shape)

class NDList:
    def __init__(self, data):
        self.ndim = len(data.shape)
        self.capacity = np.zeros(self.ndim, dtype=int)
        self.shape = self.capacity
        self.storage = np.empty(self.shape, dtype=data.dtype)
        self.data = self.storage
        self.resize(data.shape)
        self.data[:] = data
    def __getitem__(self, idcs):
        return self.data[idcs]
    def __setitem__(self, idcs, data):
        self.data[idcs] = data
    def __repr__(self):
        return 'NDList('+repr(self.data)+')'
    def __str__(self):
        return str(self.data)
    def _reserve(self, shape):
        capacity = np.stack([self.capacity, shape]).max(axis=0)
        if (capacity != self.capacity).any():
            capacity = ceil_exp_2(capacity)
            storage = np.empty(capacity, dtype=self.storage.dtype)
            return storage
        else:
            return self.storage
    def resize(self, shape):
        shape = np.array(shape)
        storage = self._reserve(shape)
        if storage is not self.storage:
            shared_shape = np.stack([shape, self.shape]).min(axis=0)
            shared_slice = tuple([slice(0,x) for x in shared_shape])
            storage[shared_slice] = self.storage[shared_slice]
            self.storage = storage
        self.shape = shape
        self.data = storage[*[slice(0,x) for x in shape]]
    def _insert_empty(self, where, expansion):
        # resizes the ndlist to prepare for insertion of data,
        # leaving unallocated regions at 'where' of size 'expansion'
        # the unallocated regions form an n-dimensional "+" shape extending in every axis
        # returns a new list of slices over the entire final shape for convenience

        # note: this could support ragged nd data with an additional parameter specifying which axes to shift old data

        lower = np.array(where)
        expansion = np.array(expansion)
        upper = lower + expansion
        old_shape = self.shape
        new_shape = self.shape + expansion
        storage = self._reserve(new_shape)

        axes_expanding_mask = (expansion != 0)
        axes_expanding_idcs = np.argwhere(axes_expanding_mask)[:,0]
        nexpanding = axes_expanding_idcs.shape[0]
        move_region_idcs = np.indices(np.full(nexpanding,2)).reshape(nexpanding,-1).T
        slicelist_src_data = [ slice(None, old_shape[idx]) for idx in range(self.ndim) ]
        slicelist_dst_data = [ slice(None, new_shape[idx]) for idx in range(self.ndim) ]
        for move_region_idx in move_region_idcs:
            slicelist_move_src = list(slicelist_src_data)
            slicelist_move_dst = list(slicelist_dst_data)
            # elements of move_region_idx are 1 for axes needing expansion
            for idx in range(nexpanding):
                axis = axes_expanding_idcs[idx]
                if move_region_idx[idx]:
                    # region axis shifts up from insertion
                    slicelist_move_src[axis] = slice(lower[axis], old_shape[axis])
                    slicelist_move_dst[axis] = slice(upper[axis], new_shape[axis])
                else:
                    # region axis is below insertion
                    slicelist_move_src[axis] = slicelist_move_dst[axis] = slice(None, lower[axis])
            storage[*slicelist_move_dst] = self.storage[*slicelist_move_src]
        self.storage = storage
        self.data = storage[*slicelist_dst_data]
        self.shape = new_shape
        return slicelist_dst_data
        
    def insert(self, axis, offset, data):
        # expands along only one dimension
        where, expansion = np.zeros([2,self.ndim],dtype=int)
        where[axis] = offset
        expansion[axis] = data.shape[axis]
        insertion_shape = self.shape.copy()
        insertion_shape[axis] = data.shape[axis]
        assert (data.shape == insertion_shape).all()
        slicelist = self._insert_empty(where, expansion)
        slicelist[axis] = slice(offset, offset + expansion[axis])
        self.data[*slicelist] = data

if __name__ == '__main__':
    ndlist1 = NDList(np.array([1,2,3]))
    assert np.allclose(ndlist1.data, [1,2,3])
    ndlist1.insert(0, 2, np.array([4,5]))
    assert np.allclose(ndlist1.data, [1,2,4,5,3])

    ndlist2 = NDList(np.array([[1,2],[3,4]]))
    assert (ndlist2.data == np.array([[1,2],[3,4]])).all()
    ndlist2.insert(0, 1, np.array([[5,6],[7,8]]))
    assert (ndlist2.data == np.array([[1,2],[5,6],[7,8],[3,4]])).all()
    ndlist2.insert(1, 1, np.array([[9,10],[11,12],[13,14],[15,16]]))
    assert (ndlist2.data == np.array([[1,9,10,2],[5,11,12,6],[7,13,14,8],[3,15,16,4]])).all()
