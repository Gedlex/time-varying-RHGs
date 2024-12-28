'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2024, Alexander Erdin (aerdin@ethz.ch), ETH Zurich
%
% This project is licensed under the MIT License.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

from .controller_base import ControllerBase
from typing import Union, Literal
import cvxpy as cp
import casadi

class EMPC(ControllerBase):
    '''Construct and solve EMPC Problem'''

    def __init__(self, sys, params, **kwargs):
        super().__init__(sys, params, **kwargs)

    def _init_problem(self, sys, params, solver: Union[Literal['cvxpy'], Literal['casadi']] = 'cvxpy'):
        # Allocate placeholder problem (cvxpy or casadi)
        if solver == 'cvxpy':
            self.prob = cp.Problem(cp.Minimize(0))
        else:
            self.prob = casadi.Opti()

    def _setup_problem(self, t = 0, x_0 = None, x_T = None, periodic = False):
        # Define decision variables (cvxpy or casadi)
        if isinstance(self.prob,cp.Problem):
            del self.prob
            self.x = cp.Variable((self.sys.n, self.params.N + 1))
            self.u = cp.Variable((self.sys.m, self.params.N))
        else:
            del self.prob
            self.prob = casadi.Opti()
            self.x = self.prob.variable(self.sys.n, self.params.N + 1)
            self.u = self.prob.variable(self.sys.m, self.params.N)            

        # Define objective
        objective = 0
        for k in range(self.params.N):
            objective += self.params.stage_cost(self.x[:,k], self.u[:,k], t=t+k)

        # Define dynamics constraints
        self.dynamics_constraints = []
        for k in range(self.params.N):
            self.dynamics_constraints += [self.x[:,k+1].reshape((self.sys.n,1)) == self.sys.f(self.x[:,k], self.u[:,k], t=t+k)]

        # Define input constraints
        self.input_constraints = [] if x_0 is None else [self.x[:, 0] == x_0]
        for k in range(self.params.N):
            self.input_constraints += [self.params.h_u(self.u[:,k], t=t+k) <= 0]

        # Define state constraints
        self.state_constraints = [] if x_0 is None else [self.x[:, 0] == x_0]
        for k in range(self.params.N+1):
            self.state_constraints += [self.params.h_x(self.x[:,k], t=t+k) <= 0]

        # Define terminal constraints
        self.state_constraints += [] if x_T is None  else [self.x[:,self.params.N] == x_T]
        self.state_constraints += [] if not periodic else [self.x[:,0] == self.x[:,self.params.N]]

        # Setup solver (cvxpy or casadi)
        if isinstance(self.x, cp.Expression):
            self.prob = cp.Problem(cp.Minimize(objective), self.dynamics_constraints + self.state_constraints + self.input_constraints)
        else:
            self.prob.minimize(objective)
            self.prob.subject_to(self.dynamics_constraints + self.state_constraints + self.input_constraints)

    def _set_parameters(self, **kwargs):
        self._setup_problem(**kwargs)

    def _output_mapping(self, output):
        if output == 'control':
            return self.u
        elif output == 'state':
            return self.x