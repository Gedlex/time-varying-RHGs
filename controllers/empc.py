'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2024, Alexander Erdin (aerdin@ethz.ch), ETH Zurich
%
% This project is licensed under the MIT License.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

from .controller_base import ControllerBase
import casadi

class EMPC(ControllerBase):
    '''Construct and solve EMPC Problem'''

    def __init__(self, sys, params):
        super().__init__(sys, params)

    def _init_problem(self, sys, params):
        # Define the EMPC problem
        self.prob = casadi.Opti()

        # Define decision variables
        self.x = self.prob.variable(sys.n, params.N + 1)
        self.u = self.prob.variable(sys.m, params.N)
        self.x_0 = self.prob.parameter(sys.n)
        self.t = self.prob.parameter(1)
        
        # Define objective
        objective = 0
        for k in range(params.N):
            objective += self._stage_cost(self.x[:,k], self.u[:,k])
        self.prob.minimize(objective)
        
        # Define constraints
        self.prob.subject_to(self.x[:,0] == self.x_0)
        for k in range(params.N):
            # Dynamics
            self.prob.subject_to(self.x[:,k+1] == sys.f(self.x[:,k], self.u[:,k], t=self.t+k))

            # State and input constraints
            self.prob.subject_to(params.h_x(self.t + k, self.x[:,k]) <= 0)
            self.prob.subject_to(params.A_u @ self.u[:,k] <= params.b_u)
        self.prob.subject_to(params.h_x(self.t + params.N, self.x[:,params.N]) <= 0)
        
        # Set NLP solver
        self.prob.solver('ipopt')

    def _stage_cost(self, x, u):
        return u.T @ u
    
    def _set_additional_parameters(self, t):
        self.prob.set_value(self.t, t)

    def _define_output_mapping(self):
        return {
            'control': self.u,
            'state': self.x,
        }

