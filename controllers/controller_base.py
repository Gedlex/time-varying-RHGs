'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2024, Alexander Erdin (aerdin@ethz.ch), ETH Zurich
%
% This project is licensed under the MIT License.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

from abc import ABC, abstractmethod
from typing import Union, Literal
import cvxpy
import casadi
import warnings

class ControllerBase(ABC):

    def __init__(self, sys, params, *args, **kwargs):
        self.sys = sys
        self.params = params
        self._init_problem(sys, params, *args, **kwargs)

    @abstractmethod 
    def _init_problem(self, sys, params, *args, **kwargs):
        '''
        This method must be implemented by the controller to define the optimization problem
        '''
        raise NotImplementedError
    
    def _set_parameters(self, **kwargs):
        '''
        Most controllers require setting parameters of the optimization problem like for example the initial condition

        This method will be called to set parameters right before calling the solver
        '''
        if kwargs:
            raise NotImplementedError(f'_set_parameters is not implemented for this controller. Received parameters: {", ".join(f"{k}={v}" for k, v in kwargs.items())}.')
        else:
            return NotImplemented
    
    @abstractmethod 
    def _output_mapping(self, output: Union[Literal['control'], Literal['state']]):
        '''
        This method must be implemented by the controller to define the mapping from the optimization
        variables to the outputs.
        '''

        ''' TEMPLATE
        return {
            'control': # planned control input trajectory,
            'state':   # planned state trajectory
        }
        '''
        raise NotImplementedError

    def solve(self, solver=None, options={}, verbose=False, **kwargs):
        if self.prob != None:
            if isinstance(self.prob,cvxpy.Problem):
                try:
                    self._set_parameters(**kwargs)
                    self.prob.solve(solver=solver, **options, verbose=verbose)

                    if self.prob.status == cvxpy.OPTIMAL:
                        error_msg = None
                        control = self._output_mapping('control').value
                        state = self._output_mapping('state').value
                        dual_values = list(self.prob.solution.dual_vars.values())
                        solver_stats = self.prob.solver_stats
                    else:
                        error_msg = 'Solver did not achieve an optimal solution. Status: {0}'.format(self.prob.status)
                        control, state, dual_values, solver_stats = (None, None, None, None)
                except Exception as e:
                    error_msg = 'Solver encountered an error. {0}'.format(e)
                    control, state, dual_values, solver_stats = (None, None, None, None)

            elif isinstance(self.prob,casadi.Opti):
                # Casadi will raise an exception if solve() detects an infeasible problem
                try :
                    self._set_parameters(**kwargs)

                    # Set options and solver
                    options = options or ({'ipopt.print_level': 5, 'print_time': 1} if verbose else {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0})
                    solver = solver or 'ipopt'
                    self.prob.solver(solver, options)
                    sol = self.prob.solve()

                    if sol.stats()['success']:
                        error_msg = None
                        control = sol.value(self._output_mapping('control'))
                        state = sol.value(self._output_mapping('state'))
                        lam_g = sol.value(self.prob.lam_g)
                        dual_values, cidx = [], 0
                        for val in self.prob.advanced.constraints():
                            n = val.numel()
                            dual_values.append(lam_g[cidx:cidx+n].reshape(-1,1))
                            cidx += n
                        solver_stats = sol.stats()
                    else:
                        error_msg = 'Solver was not successful with return status: {0}'.format(sol.stats()['return_status'])
                        control, state, dual_values, solver_stats = (None, None, None, None)
                except Exception as e:
                    error_msg = 'Solver encountered an error. {0}'.format(e)
                    control, state, dual_values, solver_stats = (None, None, None, None)

            else:
                raise Exception('Optimization problem type not supported!')
        else:
            raise Exception('Optimization problem is not initialised!')

        return control, state, error_msg, dual_values, solver_stats
