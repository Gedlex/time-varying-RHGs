'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2024, Alexander Erdin (aerdin@ethz.ch), ETH Zurich
%
% This project is licensed under the MIT License.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

from .controller_base import ControllerBase
import cvxpy as cp
import numpy as np

class CPEMPC(ControllerBase):
    '''Construct and solve EMPC Problem'''

    def __init__(self, sys, params):
        super().__init__(sys, params)

    def _init_problem(self, sys, params):
        # Define the EMPC problem
        self.prob = cp.Problem(cp.Minimize(0))
        self.x_0 = None

    def _setup_problem(self, t):
        # Clear existing problem
        del self.prob

        # Define decision variables
        self.x = cp.Variable((self.sys.n, self.params.N + 1))
        self.u = cp.Variable((self.sys.m, self.params.N))
        self.x_0 = cp.Parameter((self.sys.n))

        # Define objective
        objective = 0.0
        for k in range(self.params.N):
            objective += self.params.stage_cost(self.x[:,k], self.u[:,k], t=t+k)
        
        # Define constraints
        constraints = [self.x[:, 0] == self.x_0]
        for k in range(self.params.N):
            # Dynamics
            constraints += [self.x[:,k+1] == self.sys.f(self.x[:,k], self.u[:,k], t=t+k).reshape((self.sys.n,))]

            # State and input constraints
            constraints += [self.params.h_x(self.x[:,k], t=t+k) <= 0]
            constraints += [self.params.h_u(self.u[:,k], t=t+k) <= 0]
        constraints += [self.params.h_x(self.x[:,self.params.N], t=t+self.params.N) <= 0]

        # Setup solver
        self.prob = cp.Problem(cp.Minimize(objective), constraints)

    def _set_additional_parameters(self, t):
        self._setup_problem(t)

    def _output_mapping(self, output):
        if output == 'control':
            return self.u
        elif output == 'state':
            return self.x