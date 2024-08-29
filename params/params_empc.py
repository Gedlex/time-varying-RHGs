'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2024, Alexander Erdin (aerdin@ethz.ch), ETH Zurich
%
% This project is licensed under the MIT License.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

import numpy as np

class EMPCParams:
    
    class ctrl:
        name = 'EMPC'
        N = 10

    class sys:
        # system dimensions
        n = 1
        m = 1
        dt = 1

        # dynamics matrices
        A = 1
        B = 1
        C = 1
        D = 0

        # state constraints
        A_x = np.array([1,-1]).reshape(-1,1)
        b_x = np.array([2, 2]).reshape(-1,1)

        # input constraints
        A_u = np.array([1,-1]).reshape(-1,1)
        b_u = np.array([3, 3]).reshape(-1,1)

        # noise description
        A_w = None
        b_w = None

    class sim:
        num_steps = 30
        num_traj = 1
        x_0 = 0

    class plot:
        show = True
        color = 'red'
        alpha = 1.0
        linewidth = 1.0
