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
import casadi

class NonlinearSystem(SystemBase):
    
    def __init__(self, params):
        super().__init__(params)
        
        # Check that the nonlinear dynamics f(x, u) is defined
        if not hasattr(params, "f"):
            raise Exception("Nonlinear dynamics f(x, u) must be defined within the system parameters!")
        self._f = params.f

        # Check that nonlinear dynamics f(x, u) returns a casadi data type
        if not isinstance(self._f(casadi.MX.sym('x',self.n,1), casadi.MX.sym('u',self.m,1)), casadi.casadi.MX):            
            print("WARNING: Nonlinear dynamics function f(x, u) does not return a casadi data type.\nThis may cause issues with MPC controllers using casadi!")

        # Store the differential dynamics for the nonlinear system if they're defined in params
        if hasattr(params, "diff_A") and hasattr(params, "diff_B"):
            self.diff_A, self.diff_B = (params.diff_A, params.diff_B)

    def f(self, x, u, *args, **kwargs):
        self._check_x_shape(x)  # make sure x is n dimensional
        self._check_u_shape(u)  # make sure u is m dimensional
        return self._f(x, u, *args, **kwargs)

    def h(self, x, u, *args, **kwargs):
        self._check_x_shape(x)  # make sure x is n dimensional
        self._check_u_shape(u)  # make sure u is m dimensional
        return x