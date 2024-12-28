'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2024, Alexander Erdin (aerdin@ethz.ch), ETH Zurich
%
% This project is licensed under the MIT License.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
import os
import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.linalg import block_diag

class DSMPCParams:
    def __init__(self, T=24, **kwargs):
        self.name = 'DSMPC'

        # Define number of agents (active and passive)
        self.M = 10
        self.M_passive = 5

        # Define period length of system and constraints
        self.T = T

        # Create system and controller parameters
        self.ctrl = DSMPCParams.ctrl(self, **kwargs)
        self.sys = DSMPCParams.sys(self, **kwargs)

    class ctrl:
        def __init__(self, params, verbose=False, **kwargs):
            # Define horizon
            self.N = 24
            self.T = params.T

            # Define cost parameters
            self.gamma_1 = 5 * np.ones((params.T, params.M))
            self.gamma_2 = 0.1*np.ones((params.T, params.M))
            self.rho_1   = 1 * np.ones((params.T, params.M)) # [$/kWh]
            self.rho_2   = 0.5*np.ones((params.T, params.M)) # [$/kWh]
            for i in [6,7,8,9]:
                self.rho_1[i::12] *= 1.2
                self.rho_2[i::12] *= 2

            # Define constraint parameters
            self.q_max =  15* np.ones([params.T, params.M])
            self.zeta_max=10* np.ones([params.T, params.M])
            self.zeta_max[-1,:] = 1
            self.zeta_min = -self.zeta_max
            self.s_max  = 10.5*np.ones([params.T, params.M])
            self.s_min  =-self.s_max
            self.e_max  = 15 * np.ones([params.T, params.M]) # [kW]
            self.e_min  = 0.05*np.ones([params.T, params.M]) # [kW]
            self.l_max  = 15 * np.ones([params.T, params.M])
            self.l_min  =  0 * np.ones([params.T, params.M])
            self.L_max  =  1 * self.l_max.sum(axis=1)
            self.L_min  =  1 * self.l_min.sum(axis=1)

            # Update attributes with keyword arguments
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)

            # Load data for newyork
            data = DSMPCParams._load_data(city='newyork')

            # Define start and end date
            self.start_date = pd.Timestamp('2019-05-02 00:00:00', tz='US/Eastern')
            self.end_date = self.start_date + pd.Timedelta(hours = params.T-1)

            # Filter data
            data = data.loc[self.start_date:self.end_date, :]
            self.consumption, self.solar, self.passive_load, self.data, _ = DSMPCParams._filter_data(data, params.M, params.M_passive, remove_agents=['950', '1240'])

            # Scale down solar data (as too much solar energy is produced)
            self.solar *= 0.35

            # Compute cost matrices
            self.Q, self.R, self.c = self._compute_cost_matrices(params)

            # Compute constraint matrices
            self.X, self.c_x, self.U, self.c_u = self._compute_constraint_matrices(params)

            # Print information
            if verbose:
                print(f'Agents: {_}')
                print(f'Loaded data for {params.M} agents from {self.start_date} to {self.end_date}')
                print(f'Cost matrices: Q shape: {self.Q.shape}, R shape: {self.R.shape}, c shape: {self.c.shape}')
                print(f'Constraint matrices: X shape: {self.X.shape}, c_x shape: {self.c_x.shape}, U shape: {self.U.shape}, c_u shape: {self.c_u.shape}')
                DSMPCParams._check_convexity(Q=self.Q, R=self.R)

        # Define stage cost
        def stage_cost(self, x, u, t):
            # Wrap time index
            idx = DSMPCParams._wrap_time_index(t, self.T)
            
            if isinstance(x, cp.Expression):
                return 1/2 * cp.quad_form(x, self.Q[idx,:]) + 1/2 * cp.quad_form(u, self.R[idx,:]) + self.c[idx,:] @ u
            else:
                return 1/2 * x.T @ self.Q[idx,:] @ x + 1/2 * u.T @ self.R[idx,:] @ u + self.c[idx,:] @ u

        # State constraints
        def h_x(self, x, t):
            # Wrap time index
            idx = DSMPCParams._wrap_time_index(t, self.T)
            
            return self.X[idx,:] @ x - self.c_x[idx,:]

        # Input constraints
        def h_u(self, u, t):
            # Wrap time index
            idx = DSMPCParams._wrap_time_index(t, self.T)

            return self.U[idx,:] @ u - self.c_u[idx,:]
        
        def _compute_cost_matrices(self, params):
            # Compute cost matrices
            c = [[ np.ndarray for _ in range(params.M)] for _ in range(params.T)]
            Q = [[ np.ndarray for _ in range(params.M)] for _ in range(params.T)]
            R = [[[np.ndarray for _ in range(params.M)] for _ in range(params.M)] for _ in range(params.T)]
            for t in range(params.T):
                # Compute uncontrolled aggregate load
                agg_load = -self.solar[t,:].sum() + self.passive_load[t]

                # Loop across rows
                for i in range(params.M):
                    # Compute quadratic state cost matrix
                    Q[t][i] = 2 * np.array([[self.gamma_1[t,i], 0], [0, self.gamma_2[t,i]]])

                    # Compute linear input cost vector
                    const = self.rho_1[t,i] * agg_load - self.rho_2[t,i] * self.solar[t,i] + self.rho_2[t,i]
                    c[t][i] = const * np.ones((1, 2))

                    # Loop accros columns
                    for j in range(params.M):
                        # Compute quadratic input cost matrix
                        if i == j:
                            R[t][i][j] = 2 * self.rho_1[t,i] * np.ones((2, 2))
                        else:
                            R[t][i][j] = self.rho_1[t,i] * np.ones((2, 2))

                    # Stack columns
                    R[t][i] = np.hstack(R[t][i])

                # Stack rows
                R[t] = np.vstack(R[t])

                # Stack block diagonal matrices
                Q[t] = block_diag(*Q[t])

                # Stack vectors
                c[t] = np.hstack(c[t])

            # Stack time-varying matrices
            Q = np.stack(Q)
            R = np.stack(R)
            c = np.stack(c)
            return Q, R, c

        def _compute_constraint_matrices(self, params):
            # Compute state constraints
            X = [[np.ndarray for _ in range(params.M)] for _ in range(params.T)]
            c_x = [[np.ndarray for _ in range(params.M)] for _ in range(params.T)]
            for t in range(params.T):
                for v in range(params.M):
                    X[t][v] = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
                    c_x[t][v] = np.array([self.zeta_max[t,v], -self.zeta_min[t,v], self.q_max[t,v], 0])

                # Stack constraints in a block diagonal matrix
                X[t] = block_diag(*X[t])
                c_x[t] = np.hstack(c_x[t])
            
            # Stack time-varying matrices
            X = np.stack(X)
            c_x = np.stack(c_x)

            # Compute input constraints
            U = [[np.ndarray for _ in range(params.M)] for _ in range(params.T)]
            c_u = [[np.ndarray for _ in range(params.M)] for _ in range(params.T)]
            for t in range(params.T):
                for v in range(params.M):
                    U[t][v] = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1]])
                    c_u[t][v] = np.array([self.e_max[t,v], -self.e_min[t,v], self.s_max[t,v], -self.s_min[t,v], self.l_max[t,v] + self.solar[t,v], -self.l_min[t,v] - self.solar[t,v]])

                # Stack constraints in a block diagonal matrix
                U[t] = block_diag(*U[t])
                c_u[t] = np.hstack(c_u[t])

                # Add coupling constraints / aggregate load limit
                agg_load = -self.solar[t,:].sum() + self.passive_load[t]
                U[t] = np.vstack([U[t], np.ones((1, 2*params.M)), -np.ones((1, 2*params.M))])
                c_u[t] = np.hstack([c_u[t], np.array([self.L_max[t] - agg_load, -self.L_min[t] + agg_load])])
            
            # Stack time-varying matrices
            U = np.stack(U)
            c_u = np.stack(c_u)
            return X, c_x, U, c_u

    class sys:
        def __init__(self, params, verbose=False, **kwargs):
            # Define system dimensions
            self.n = 2 * params.M
            self.m = 2 * params.M
            self.dt = 1
            self.T = params.T

            # Define constants
            alpha = 0.9**(1/params.T) * np.ones((params.T, params.M))
            beta  = 0.9 * np.ones((params.T, params.M))

            # Define system dynamics
            self.A = np.stack([block_diag(*[np.array([[1, 0], [0, alpha[t,v]]]) for v in range(params.M)]) for t in range(params.T)])
            self.B = np.stack([block_diag(*[np.array([[1, 0], [0,  beta[t,v]]]) for v in range(params.M)]) for t in range(params.T)])
            self.d = np.stack([ np.vstack( [np.array([[-params.ctrl.consumption[t,v]], [0]]) for v in range(params.M)]) for t in range(params.T)])
            self.C = np.stack([ np.eye(self.n, self.n) for t in range(params.T)])
            self.D = np.zeros((self.T, self.n, self.m))

            # Print information
            if verbose:
                DSMPCParams._check_invertibility(A=self.A)
        
    @staticmethod
    def _load_data(city='newyork'):
        # Get class directory
        dir = os.path.dirname(os.path.abspath(__file__))   

        # Construct filepath
        filepath = os.path.join(dir, f'processed_data_{city}.csv')

        # Load data
        data = pd.read_csv(filepath, engine='python',
                           encoding="ISO-8859-1", header = [0, 1], index_col=0)

        # Convert index to datetime objects
        data.index = pd.to_datetime(data.index, utc=False)
        return data
    
    @staticmethod
    def _filter_data(data, num_agents, num_passive_agents, remove_agents = []):            
        # Get all agent ids, excluding those in remove_agents
        agent_ids = [agent for agent in data.columns.get_level_values(1).unique() if agent not in remove_agents]

        # Select the first `num_agents` from agent_ids
        selected_agents = agent_ids[:num_agents]

        # Filter data for the date range and selected agents
        filtered_data = data.loc[:, pd.IndexSlice[:, selected_agents]]

        # Get consumption data
        consumption = filtered_data.consumption.to_numpy()

        # Compute passive load
        passive_load = consumption[:, 0:num_passive_agents].sum(axis=1)

        # Get solar data
        solar = filtered_data.all_solar[:].to_numpy()

        return consumption, solar, passive_load, filtered_data, selected_agents
    
    @staticmethod
    def _wrap_time_index(t, T):
        if isinstance(t, int):
            return t % T
        elif T > 1:
            raise ValueError(f'Invalid time index: {t} of type {type(t)}. Please provide an integer time index for time-varying systems.')
        return 0
    
    @staticmethod
    def _check_convexity(**kwargs):
        results = {}
        for key, value in kwargs.items():
            # Check positive semi-definiteness for each time t
            check = np.array([DSMPCParams._is_pos_semi_def(value[k,:]) for k in range(value.shape[0])])

            # Print results
            if np.all(check):
                print(f'{key} is positive semi-definite for all times.')
            else:
                print(f'\033[93mWarning: {key} is not positive semi-definite at times t = {np.where(~check)[0]}.\033[0m')

            # Add to dictionary
            results[key] = check
        return results
    
    @staticmethod
    def _check_invertibility(**kwargs):
        results = {}
        for key, value in kwargs.items():
            # Check invertibility for each time t
            check = np.array([not np.isclose(np.linalg.det(value[k,:]), 0) for k in range(value.shape[0])])

            # Print results
            if np.all(check):
                print(f'{key} is invertible for all times.')
            else:
                print(f'\033[93mWarning: {key} is not invertible at times t = {np.where(~check)[0]}.\033[0m')

            # Add to dictionary
            results[key] = check
        return results

    @staticmethod
    def _is_pos_semi_def(A):
        if np.array_equal(A, A.T):
            try:
                np.linalg.cholesky(A + 1E-14*np.eye(A.shape[0]))
                return True
            except np.linalg.LinAlgError:
                return False
        else:
            return ValueError('Matrix is not symmetric')