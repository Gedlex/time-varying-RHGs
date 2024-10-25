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
        self.x_0 = None

    def _setup_problem(self, t):
        # Define the EMPC problem
        del self.prob
        self.prob = casadi.Opti() # Check if this is needed

        # Define decision variables
        self.x = self.prob.variable(self.sys.n, self.params.N + 1)
        self.u = self.prob.variable(self.sys.m, self.params.N)
        self.x_0 = self.prob.parameter(self.sys.n)

        # Define objective
        objective = 0
        for k in range(self.params.N):
            objective += self.params.stage_cost(self.x[:,k], self.u[:,k], t=t+k)
        self.prob.minimize(objective)

        # Define constraints
        self.prob.subject_to(self.x[:,0] == self.x_0)
        for k in range(self.params.N):
            # Dynamics
            self.prob.subject_to(self.x[:,k+1] == self.sys.f(self.x[:,k], self.u[:,k], t=t+k))

            # State and input constraints
            self.prob.subject_to(self.params.h_x(self.x[:,k], t=t+k) <= 0)
            self.prob.subject_to(self.params.h_u(self.u[:,k], t=t+k) <= 0)
        self.prob.subject_to(self.params.h_x(self.x[:,self.params.N], t=t+self.params.N) <= 0)

        # Setup NLP solver
        self.prob.solver('ipopt', {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0})

    def _set_additional_parameters(self, t):
        self._setup_problem(t)

    def _output_mapping(self, output):
        if output == 'control':
            return self.u
        elif output == 'state':
            return self.x