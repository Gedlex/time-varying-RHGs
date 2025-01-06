'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2024, Alexander Erdin (aerdin@ethz.ch), ETH Zurich
%
% This project is licensed under the MIT License.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

from .controller_base import ControllerBase
from types import MethodType
import numpy as np
import cvxpy as cp
import casadi

class CEMPC(ControllerBase):
    '''Construct and solve CEMPC Problem'''

    def __init__(self, sys, params, **kwargs):
        super().__init__(sys, params, **kwargs)

    def _init_problem(self, sys, params):
        # Define decision variables
        self.x = cp.Variable((self.sys.n, self.params.N + 1))
        self.u = cp.Variable((self.sys.m, self.params.N))

        # Convexify stage cost
        self._convexify_stage_cost()

        # Define objective
        objective = 0
        for k in range(self.params.N):
            objective += self.params.stage_cost(self.x[:,k], self.u[:,k], t=k)

        # Define dynamics constraints
        self.dynamics_constraints = []
        for k in range(self.params.N):
            self.dynamics_constraints += [self.x[:,k+1].reshape((self.sys.n,1)) == self.sys.f(self.x[:,k], self.u[:,k], t=k)]

        # Define input constraints
        self.input_constraints = []
        for k in range(self.params.N):
            self.input_constraints += [self.params.h_u(self.u[:,k], t=k) <= 0]

        # Define state constraints
        self.state_constraints = []
        for k in range(self.params.N+1):
            self.state_constraints += [self.params.h_x(self.x[:,k], t=k) <= 0]
        self.state_constraints += [self.x[:,0] == self.x[:,self.params.N]]

        # Setup solver
        self.prob = cp.Problem(cp.Minimize(objective), self.dynamics_constraints + self.state_constraints + self.input_constraints)

    def _convexify_stage_cost(self):
        # Define decision variables
        K = [cp.Variable((self.sys.n, self.sys.n), symmetric=True) for _ in range(self.params.T)]

        # Construct LMIs
        LMIs = self._construct_LMIs(K)

        # Define objective
        if self.params.T == 1:
            objective = cp.lambda_min(LMIs[0])
        else:
            objective = cp.minimum(*[cp.lambda_min(LMI) for LMI in LMIs])

        # Define constraints
        constraints = [] # [LMI[:self.sys.n,:self.sys.n] >> 1E-8 * np.eye(self.sys.n) for LMI in LMIs]

        # Setup solver
        prob = cp.Problem(cp.Maximize(objective), constraints)

        # Solve problem
        prob.solve(solver='SCS', max_iters=10000, eps=1E-5, verbose=False)
        print(f"Convexification problem solved with minimal eigenvalue: {prob.value}")

        # Get solution
        if prob.status == cp.OPTIMAL:
            K_sol = [K_t.value for K_t in K]
            self.LMIs_sol = self._construct_LMIs(K_sol)

            # Save convexified cost matrices to params
            self.params.K = np.stack(K_sol)
            self.params.M = np.stack(self.LMIs_sol)
            self.params.M_x  = self.params.M[:,:self.sys.n,:self.sys.n]
            self.params.M_u  = self.params.M[:,-self.sys.m:,-self.sys.m:]
            self.params.M_xu = self.params.M[:,:self.sys.n,-self.sys.m:]

            # Check convexity
            self._check_convexity(LMIs=self.LMIs_sol, M_x=self.params.M_x, M_u=self.params.M_u)

            # Save convexified stage cost and gradient to params
            self.params.stage_cost = MethodType(CEMPC._convexified_stage_cost, self.params)
            self.params.stage_cost_grad = MethodType(CEMPC._convexified_stage_cost_grad, self.params)
        else:
            raise RuntimeError(f'Convexification failed. Status: {prob.status}')

    def _construct_LMIs(self, K):
        # Allocate LMIs
        LMIs = []
        is_cvxpy = isinstance(K[0], cp.Expression)
        for t, K_t in enumerate(K):
            # Get next K in periodic sequence
            K_next = K[(t + 1) % len(K)]

            # Construct LMI
            LMI = [[self.params.Q[t] / 2 + K_t - self.sys.A[t].T @ K_next @ self.sys.A[t],                      -self.sys.A[t].T @ K_next @ self.sys.B[t]],
                   [                           - self.sys.B[t].T @ K_next @ self.sys.A[t], self.params.R[t] / 2 -self.sys.B[t].T @ K_next @ self.sys.B[t]]]

            # Add LMI
            LMIs.append(cp.bmat(LMI) if is_cvxpy else np.block(LMI))
        return LMIs
    
    @staticmethod
    def _convexified_stage_cost(obj, x, u, t):
        # Wrap time index
        idx = t % obj.T

        # Stack state and control
        if isinstance(x, cp.Expression):
            z = cp.hstack([x, u])
        elif isinstance(x, casadi.casadi.MX):
            z = casadi.vertcat(x, u)
        elif isinstance(x, np.ndarray):
            z = np.hstack([x, u])
        else:
            raise ValueError(f"Invalid input type to convexified stage cost. Expected cp.Expression, casadi.MX, or np.ndarray, got {type(x)}.")

        # Return stage cost
        if isinstance(z, cp.Expression):
            return cp.quad_form(z, obj.M[idx,:]) + obj.c[idx,:] @ u
        else:
            return z.T @ obj.M[idx,:] @ z + obj.c[idx,:] @ u

    @staticmethod    
    def _convexified_stage_cost_grad(obj, x, u, t):
        # Wrap time index
        idx = t % obj.T

        # Stack state and control
        if not isinstance(x, np.ndarray):
            raise ValueError(f"Invalid input type to convexified stage cost. Expected cp.Expression, casadi.MX, or np.ndarray, got {type(x)}.")

        # Compute gradient
        grad = 2 * obj.M[idx,:] @ np.hstack([x, u])
        grad_x, grad_u = np.split(grad, [x.shape[0]])
        return grad_x, grad_u + obj.c[idx,:].reshape(-1)

    @staticmethod
    def _check_convexity(**kwargs) -> dict:
        results = {}
        for key, value in kwargs.items():
            # Check positive definiteness for all times
            check = np.array([CEMPC._is_pos_def(value[k]) for k in range(len(value))])

            # Print results
            if np.all(check):
                print(f'{key} is positive definite for all times.')
            elif np.any(check):
                raise RuntimeError(f'{key} is not positive definite at times t = {np.where(~check)[0]}')
            else:
                raise RuntimeError(f'{key} is not positive definite for all times.')

            # Add to dictionary
            results[key] = check
        return results

    @staticmethod
    def _is_pos_def(A):
        if np.allclose(A, A.T):
            try:
                np.linalg.cholesky(A)
                return True
            except np.linalg.LinAlgError:
                return False
        else:
            raise ValueError('Matrix is not symmetric')
        
    def _output_mapping(self, output):
        if output == 'control':
            return self.u
        elif output == 'state':
            return self.x