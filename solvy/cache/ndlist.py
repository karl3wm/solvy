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
    def _reserve(self, shape):
        capacity = np.stack([self.capacity, shape]).max(axis=0)
        if (capacity != self.capacity).any():
            capacity = ceil_exp_2(capacity)
            storage = np.empty(capacity, dtype=self.storage.dtype)
            return storage
        else:
            return self.storage
            #data = storage[*[slice(0,x) for x in self.shape]]
            #data[:] = self.data
            #self.data = data
            #self.storage = storage
            #self.capacity = capacity
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
        #assert (self.data.shape == shape).all()
    def insert_empty(self, where, expansion):
        # resizes the ndlist to prepare for insertion of data,
        # leaving unallocated regions between lower and upper
        lower = np.array(where)
        expansion = np.array(expansion)
        upper = lower + expansion
        old_shape = self.shape
        new_shape = self.shape + expansion
        #self.resize(new_shape)
        storage = self._reserve(new_shape)
        # ok nd expansion.
        # it has a simple pattern, which i don't know yet but would emerge.
        # we've done the 1-dimensional case.
        # formed a slice for all the preceding data, and all the following data

            # so we're inserting a hyperplus shape -- it has a hypercube extending
            # in every axis along which upper != lower
            # consider the 2D case -- there are 3 moves performed, one for each
            # quadrant involved in insertion.
            # it can also be thought of as 2 successive insertions.
            # one along one axis -- moving a single large block of data
            # and a second another another axis -- another single large block
            # alternatively, it's all the [0,1] combinations of the axes.
            # if there's a 0, the region is prior to the insert.
            # if there's a 1, the region is beyond the insert.
        axes_expanding_mask = (expansion != 0)
        axes_expanding_idcs = np.argwhere(axes_expanding_mask)[:,0]
        nexpanding = axes_expanding_idcs.shape[0]
        move_region_idcs = np.indices(np.full(nexpanding,2)).reshape(nexpanding,-1).T
        #slices = SLICE_ALL * self.ndim
        slicelist_src_data = [
            slice(None, old_shape[idx])
            for idx in range(self.ndim)
        ]
        slicelist_dst_data = [
            slice(None, new_shape[idx])
            for idx in range(self.ndim)
        ]
        for move_region_idx in move_region_idcs:
            slicelist_move_src = list(slicelist_src_data)
            slicelist_move_dst = list(slicelist_dst_data)
            # elements of move_region_idx are 1 for axes needing expansion
            # no longer: # the first is skipped as it is all 0s. it's the region below lower
                # potential problem
                # consider + shape; in a [1,0] or [0,1] quad we want the
                # [0,lower] range, but
                # consider | or -- shape. here along one axis we want the
                # _whole_ range, to move the expanded half. but that's
                # not [0,lower]; it's [0,None] == [lower,upper]
                # so there are 3 options: nonincluded, lower, and upper0
                    # can check with axes_expanding
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
        slicelist = self.insert_empty(where, expansion)
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
