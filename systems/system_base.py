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
        # Set system dimensions
        self.n = params.n   # state dimension
        self.m = params.m   # input dimension
        
        # Set time step
        self.dt = params.dt        
        
    def step(self, x, u, t=None, w=None):
        '''Advance system from state x with input u, adding a noise/disturbance'''
        x_next = self.f(x, u) if t is None else self.f(x, u, t=t)

        # Make sure that x_next is a numpy array
        if not isinstance(x_next, np.ndarray):
            x_next = np.array(x_next)

        # Add noise / disturbance
        if w is not None:
            # Make sure that w is a numpy array
            if not isinstance(w, np.ndarray):
                w = np.array(w)

            # Check shape of w
            self._check_x_shape(w)
            x_next += w

        return x_next

    def get_output(self, x, u, t=None):
        '''Evaluate output function for state x and input u'''
        output = self.h(x, u) if t is None else self.h(x, u, t=t)

        # Make sure that output is a numpy array
        if not isinstance(output, np.ndarray):
            output = np.array(output)

        return output

    @abstractmethod
    def f(self, x, u, *args, **kwargs):
        '''Nominal system update function to be implemented by the inherited class'''
        raise NotImplementedError

    @abstractmethod
    def h(self, x, u, *args, **kwargs):
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