'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2023, Alexandre Didier, Jérôme Sieber, Rahel Rickenbach and Shao (Mike) Zhang, ETH Zurich,
% {adidier,jsieber, rrahel}@ethz.ch
%
% All rights reserved.
%
% This code is only made available for students taking the advanced MPC 
% class in the fall semester of 2023 (151-0371-00L) and is NOT to be 
% distributed.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

from .system_base import SystemBase
import numpy as np
import casadi

class NonlinearSystem(SystemBase):

    def __init__(self, params):
        self.update_params(params)

    def update_params(self, params):
        super().update_params(params)

        if not hasattr(params, "f"):
            raise Exception("Nonlinear dynamics f(x, u) must be defined within the system parameters!")
        self._f = params.f

        if type(self._f(np.zeros((self.n,1)), np.zeros((self.m,1)))) not in [casadi.DM, casadi.MX, casadi.SX]:
            print("WARNING: Nonlinear dynamics function f(x, u) does not return a casadi data type.\nThis may cause issues with MPC controllers using casadi!")

        # store the differential dynamics for the nonlinear system if they're defined in params
        if hasattr(params, "diff_A") and hasattr(params, "diff_B"):
            self.diff_A, self.diff_B = (params.diff_A, params.diff_B)

    def f(self, x, u):
        self._check_x_shape(x)  # make sure x is n dimensional
        self._check_u_shape(u)  # make sure u is m dimensional
        return self._f(x, u)

    def h(self, x, u):
        self._check_x_shape(x)  # make sure x is n dimensional
        self._check_u_shape(u)  # make sure u is m dimensional
        return x
