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

        # Check shape of system matrices
        assert params.A.shape[-2::] == (self.n, self.n), 'A must have shape (,n,n)'
        assert params.B.shape[-2::] == (self.n, self.m), 'B must have shape (t,n,m)'
        assert params.C.shape[-1] == self.n, 'C must have shape (t, num_output, n)'
        assert params.D.shape[-1] == self.m, 'D must have shape (t, num_output, m)'

        # Determine if system is time-varying
        self.A = params.A.reshape(-1, self.n, self.n)
        self.T = self.A.shape[0]
        self.time_varying = self.T > 1

        # Reshape system matrices
        self.B = params.B.reshape(self.T, self.n, self.m)
        self.C = params.C.reshape(self.T, params.C.shape[-2], self.n)
        self.D = params.D.reshape(self.T, params.D.shape[-2], self.m)
        
        # Check shape of opional offset vector
        if hasattr(params, 'd'):
            assert params.d.size == self.T * self.n, 'd must have shape (t, n,)'
            self.d = params.d.reshape(self.T, self.n, 1)
        else:
            self.d = np.zeros((self.T, self.n, 1))

    def f(self, x, u, t=None):
        self._check_x_shape(x)  # make sure x is n dimensional
        self._check_u_shape(u)  # make sure u is m dimensional

        # Get system matrices at time t
        A_t, B_t, d_t = self._get_system_matrices(t, self.A, self.B, self.d)

        return A_t @ x.reshape((self.n, 1)) + B_t @ u.reshape((self.m, 1)) + d_t

    def h(self, x, u, t=None):
        self._check_x_shape(x)  # make sure x is n dimensional
        self._check_u_shape(u)  # make sure u is m dimensional

        # Get system matrices at time t
        C_t, D_t = self._get_system_matrices(t, self.C, self.D)
        
        return C_t @ x.reshape(self.n, 1) + D_t @ u.reshape(self.m, 1)
    
    def _wrap_time_index(self, t):
        if isinstance(t, (casadi.MX, casadi.SX)):
            return casadi.remainder(t, self.T)
        
        elif isinstance(t, int):
            return t % self.T
        
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