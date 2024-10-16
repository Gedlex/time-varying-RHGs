'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2024, Alexander Erdin (aerdin@ethz.ch), ETH Zurich
%
% This project is licensed under the MIT License.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

import numpy as np
import casadi

class EMPCParamsGruene:

    class ctrl:
        name = 'EMPC'
        # Horizon
        N = 8

        # Define stage cost
        @staticmethod
        def stage_cost(x, u):
            return u.T @ u

        # State constraints
        @staticmethod
        def h_x(t, x):
            condition = casadi.fmod(t, 24) < 12
            constraint1 = np.array([1, -1]).reshape(-1, 1) @ x - np.array([2, 2]).reshape(-1, 1)
            constraint2 = np.array([1, -1]).reshape(-1, 1) @ x - np.array([1/2, 1/2]).reshape(-1, 1)
            
            return casadi.if_else(condition, constraint1, constraint2)

        # Input constraints
        def h_u(t, u):
            return np.array([1,-1]).reshape(-1,1) @ u - np.array([3, 3]).reshape(-1,1)

    class sys:
        # System dimensions
        n = 1
        m = 1
        dt = 1

        # Nonlinear dynamics
        def f(x, u, t=0):
            # Generate a random disturbance
            w = -2*np.sin(t*np.pi/12) + (0.5*np.random.rand(1) - 0.25)

            # Return the next state
            return x + u + w
        
    class sim:
        num_steps = 50
        num_traj = 10
        x_0 = np.linspace(-2,2,10)

    class plot:
        show = True
        color = 'red'
        alpha = 1.0
        linewidth = 1.0
