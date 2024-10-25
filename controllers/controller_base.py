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

    @abstractmethod
    def _set_additional_parameters(self, additional_parameters):
        '''
        Some controllers require setting additional parameters of the optimization problem beside just setting the initial condition

        For controllers which require additional parameters, they must override this method
        to set the value of those parameters

        This method will be called to set the additional parameters right before calling the solver
        '''
        return NotImplementedError
    
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

    def solve(self, x, additional_parameters=None, verbose=False, solver=None):
        if self.prob != None:
            if not hasattr(self, 'x_0'):
                raise Exception(
                    'The MPC problem must define the initial condition as an optimization parameter self.x_0')
            
            if isinstance(self.prob,cvxpy.Problem):
                try:
                    if additional_parameters is not None:
                        self._set_additional_parameters(additional_parameters)
                    self.x_0.value = x
                    self.prob.solve(verbose=verbose, solver=solver)

                    if self.prob.status != cvxpy.OPTIMAL:
                        error_msg = 'Solver did not achieve an optimal solution. Status: {0}'.format(self.prob.status)
                        control, state = (None, None)
                    else:
                        error_msg = None
                        control = self._output_mapping('control').value
                        state = self._output_mapping('state').value
                except Exception as e:
                    error_msg = 'Solver encountered an error. {0}'.format(e)
                    control, state = (None, None)

            elif isinstance(self.prob,casadi.Opti):
                if verbose:
                    opts = {'ipopt.print_level': 5, 'print_time': 1}
                else:
                    opts = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
                self.prob.solver('ipopt', opts)

                # Casadi will raise an exception if solve() detects an infeasible problem
                try:
                    if additional_parameters is not None:
                        self._set_additional_parameters(additional_parameters)
                    self.prob.set_value(self.x_0, x)
                    sol = self.prob.solve()
                    if sol.stats()['success']:
                        error_msg = None
                        control = sol.value(self._output_mapping('control'))
                        state = sol.value(self._output_mapping('state'))

                    else:
                        error_msg = 'Solver was not successful with return status: {0}'.format(sol.stats()['return_status'])
                        control, state = (None, None)
                except Exception as e:
                    error_msg = 'Solver encountered an error. {0}'.format(e)
                    control, state = (None, None)

            else:
                raise Exception('Optimization problem type not supported!')
        else:
            raise Exception('Optimization problem is not initialised!')

        return control, state, error_msg
