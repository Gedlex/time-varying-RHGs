'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2024, Alexander Erdin (aerdin@ethz.ch), ETH Zurich
%
% This project is licensed under the MIT License.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

from abc import ABC, abstractmethod
import numpy as np

class SystemBase(ABC):

    def __init__(self, params):
        self.update_params(params)

    def update_params(self, params):
        # time step
        self.dt = params.dt

        # system dimensions
        self.n = params.n
        self.m = params.m

        # systems constraints as polytopes
        if params.A_x is not None and params.b_x is not None:
            self.X = Polytope(params.A_x, params.b_x)

        if params.A_u is not None and params.b_u is not None:
            self.U = Polytope(params.A_u, params.b_u)

        if params.A_w is not None and params.b_w is not None:
            self.W = Polytope(params.A_w, params.b_w)
        
    def step(self, x, u, w=None):
        '''Advance system from state x with input u, adding a noise/disturbance'''
        x_next = self.f(x, u)

        # make sure that x_next is a numpy array
        if not isinstance(x_next, np.ndarray):
            x_next = np.array(x_next)

        # add noise / disturbance
        if w is not None:
            self._check_x_shape(w)
            x_next += w

        return x_next

    def get_output(self, x, u):
        '''Evaluate output function for state x and input u'''
        output = self.h(x, u)

        # make sure that output is a numpy array
        if not isinstance(output, np.ndarray):
            output = np.array(output)

        return output

    @classmethod
    @abstractmethod
    def f(self, x, u):
        '''Nominal system update function to be implemented by the inherited class'''
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def h(self, x, u):
        '''Nominal system output function to be implemented by the inherited class'''
        raise NotImplementedError

    def _check_x_shape(self, x):
        '''
        Verifies the shape of x
        Usable if x is a float or an array type which defines a shape property (e.g. numpy and casadi arrays)
        '''
        if hasattr(x, 'shape') and self.n > 1:
            assert x.shape == (self.n, 1) or x.shape == (self.n,), 'x must be {0} dimensional, instead has shape {1}'.format(self.n, x.shape)

    def _check_u_shape(self, u):
        '''
        Verifies the shape of u
        Usable if u is a float or an array type which defines a shape property (e.g. numpy and casadi arrays)
        '''
        if hasattr(u, 'shape') and self.m > 1:
            assert u.shape == (self.m, 1) or u.shape == (self.m,), 'u must be {0} dimensional, instead has shape {1}'.format(self.m, u.shape)

class Polytope():
    def __init__(self, A, b):
        assert A.shape[0] == b.shape[0], 'A and b must have the same number of rows'
        self.A = A
        self.b = b

    def contains(self, x):
        return np.all(self.A @ x <= self.b)

    def __str__(self):
        return f"Polytope(A={self.A}, b={self.b})"

    def __repr__(self):
        return self.__str__()
