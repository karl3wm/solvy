import poly

import numpy as np

class Equations:
    def __init__(self, func, in_minmaxes : dict, out_minmaxes : dict):
        self.ins = list(input_extrema.keys())
        self.outs = list(output_extrema.keys())
        self.vars = input_extrema | output_extrema
        self.func = func
        self._data = []
    def __call__(self, **inputs):
        return self.func(**inputs)
    def solve(self, new_inputs):
        # we will construct a piecewise polynomial function that is the solution to this function.
        # we can start by collecting data over the bounds.

            # interestingly we're basically casting the variable space as subdivisions of nd space, kind of binary subdivisions. that can become a lot of subdivisions
            # - we only need to consider the space of the new inputs.
            # - we only need to subdivide variables that change the output a lot

