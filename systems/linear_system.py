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
import cvxpy
import casadi

class LinearSystem(SystemBase):
    
    def __init__(self, params):
        super().__init__(params)

        # Check that system matrices have the correct shape
        assert params.A.shape[-2::] == (self.n, self.n), 'A must have shape (n,n)'
        assert params.B.shape[-2::] == (self.n, self.m), 'B must have shape (n,m)'
        assert params.C.shape[-1] == self.n, 'C must have shape (num_output, n)'
        assert params.D.shape[-1] == self.m, 'D must have shape (num_output, m)'
        self.A = params.A.reshape(-1, self.n, self.n)
        self.B = params.B.reshape(-1, self.n, self.m)
        self.C = params.C.reshape(-1, params.C.shape[-2], self.n)
        self.D = params.D.reshape(-1, params.D.shape[-2], self.m)

        # Determine if system is time-varying
        self.time_varying = self.A.shape[0] > 1

    def f(self, x, u, t=None):
        self._check_x_shape(x)  # make sure x is n dimensional
        self._check_u_shape(u)  # make sure u is m dimensional

        # Get system matrices at time t
        A, B = self._get_system_matrices(t, self.A, self.B)

        return A @ x.reshape(self.n, 1) + B @ u.reshape(self.m, 1)

    def h(self, x, u, t=None):
        self._check_x_shape(x)  # make sure x is n dimensional
        self._check_u_shape(u)  # make sure u is m dimensional

        # Get system matrices at time t
        C, D = self._get_system_matrices(t, self.C, self.D)
        
        return C @ x.reshape(self.n, 1) + D @ u.reshape(self.m, 1)
    
    def _wrap_time_index(self, t):
        if isinstance(t, (casadi.MX, casadi.SX)):
            return casadi.remainder(t, self.A.shape[0])
        
        elif isinstance(t, int):
            return t % self.A.shape[0]
        
        elif isinstance(t, cvxpy.Parameter):
            raise NotImplementedError("Time indexing with cvxpy parameters is not supported yet.")
        else:
            raise TypeError(f"Unsupported type {type(t)} for time index t.")
        
    def _get_system_matrices(self, t, *args):
        # Get system matrices at time t
        if self.time_varying:
            if t is None:
                raise ValueError('Time index t must be provided for time-varying system matrices')
            else:
                # Wrap time index around
                t = self._wrap_time_index(t)

                # Check if t is a casadi parameter
                if isinstance(t, (casadi.MX, casadi.SX)):
                    # Compute one-hot encoded system matrices
                    Z_t = [np.zeros(Z.shape[1::]) for Z in args]
                    for i in range(args[0].shape[0]):
                        for j in range(len(args)):
                            Z_t[j] += casadi.if_else(t == i, args[j][i,::], 0)
                
                # Check if t is an integer
                elif isinstance(t, int):
                    # Get system matrices at time t
                    Z_t = [Z[t,::] for Z in args]
        else:
            # Get time-invariant system matrices
            Z_t = [Z[0,::] for Z in args]

        return Z_t