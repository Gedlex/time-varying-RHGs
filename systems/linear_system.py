'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2024, Alexander Erdin (aerdin@ethz.ch), ETH Zurich
%
% This project is licensed under the MIT License.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

from .system_base import SystemBase
import numpy as np

class LinearSystem(SystemBase):
    
    def __init__(self, params):
        super().__init__(params)

        # Check shape of system matrices
        assert params.A.shape[-2:] == (self.n, self.n), 'A must have shape (t,n,n)'
        assert params.B.shape[-2:] == (self.n, self.m), 'B must have shape (t,n,m)'
        assert params.C.shape[-1] == self.n, 'C must have shape (t, num_output, n)'
        assert params.D.shape[-1] == self.m, 'D must have shape (t, num_output, m)'

        # Determine if system is time-varying
        self.T = params.A.shape[0] if len(params.A.shape) > 2 else 1
        self.time_varying = self.T > 1

        # Reshape system matrices
        self.A = params.A.reshape(self.T, self.n, self.n)
        self.B = params.B.reshape(self.T, self.n, self.m)
        self.C = params.C.reshape(self.T, params.C.shape[-2], self.n)
        self.D = params.D.reshape(self.T, params.D.shape[-2], self.m)
        
        # Get optional offset vector
        self.d = getattr(params, 'd', np.zeros((self.T, self.n))).reshape(self.T, self.n, 1)

    def f(self, x, u, t=None):
        self._check_x_shape(x)  # make sure x is n dimensional
        self._check_u_shape(u)  # make sure u is m dimensional
        idx = self._wrap_time_index(t) # Wrap time index
        return self.A[idx,:] @ x.reshape((self.n, 1)) + self.B[idx,:] @ u.reshape((self.m, 1)) + self.d[idx,:]
    
    def h(self, x, u, t=None):
        self._check_x_shape(x)  # make sure x is n dimensional
        self._check_u_shape(u)  # make sure u is m dimensional
        idx = self._wrap_time_index(t) # Wrap time index
        return self.C[idx,:] @ x.reshape((self.n, 1)) + self.D[idx,:] @ u.reshape((self.m, 1))
    
    def f_grad(self, *args, t=None):
        idx = self._wrap_time_index(t)
        return self.A[idx,:].T, self.B[idx,:].T
    
    def _wrap_time_index(self, t):
        if isinstance(t, int):
            return t % self.T
        elif self.time_varying:
            raise ValueError(f'Invalid time index: {t} of type {type(t)}. Please provide an integer time index for time-varying systems.')
        return 0