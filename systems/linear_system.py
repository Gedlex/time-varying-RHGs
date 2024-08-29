'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2024, Alexander Erdin (aerdin@ethz.ch), ETH Zurich
%
% This project is licensed under the MIT License.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

from .system_base import SystemBase

class LinearSystem(SystemBase):

    def __init__(self, params):
        self.update_params(params)

    def update_params(self, params):
        super().update_params(params)
        assert params.A.shape == (self.n, self.n), 'A must have shape (n,n)'
        assert params.B.shape == (self.n, self.m), 'B must have shape (n,m)'
        assert params.C.shape[1] == self.n, 'C must have shape (num_output, n)'
        assert params.D.shape[1] == self.m, 'D must have shape (num_output, m)'
        self.A = params.A
        self.B = params.B
        self.C = params.C
        self.D = params.D

    def f(self, x, u):
        self._check_x_shape(x)  # make sure x is n dimensional
        self._check_u_shape(u)  # make sure u is m dimensional
        return self.A @ x.reshape(self.n, 1) + self.B @ u.reshape(self.m, 1)

    def h(self, x, u):
        self._check_x_shape(x)  # make sure x is n dimensional
        self._check_u_shape(u)  # make sure u is m dimensional
        return self.C @ x.reshape(self.n, 1) + self.D @ u.reshape(self.m, 1)

