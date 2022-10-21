# version 1.0

import numpy as np
import sys
from typing import List, Dict

from utils_soln import *

e = Environment(2,[1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1],2,[[0.8, 0.2], [0.1, 0.9]], 
                [{0:0.1, -1:0.8, -2:0.1},{0:0.1, 1:0.8, 2:0.1},{0:1.0}], None)

e1 = Environment(2,[0,1],2,[[0.9, 0.1],[0.2, 0.8]], None, [[[0.7, 0.3],[0.3, 0.7]]])

e2 = Environment(2, [1,0], 2, [[0.8, 0.2], [0.1, 0.9]], [{0:0.05, -1:0.95}, {0:0.15, 1:0.85}, {0:1.0}], None)

e3 = Environment(2, [1,0], 2, [[0.8, 0.2], [0.1, 0.9]], [{0:0.05, -1:0.95}, {0:0.15, 1:0.85}, {0:1.0}], None)

observations = [1,0,0]
actions = [1,0]
probs = [0.5, 0.5]


def create_observation_matrix(env: Environment):
    '''
    Creates a 2D numpy array containing the observation probabilities for each state. 

    Entry (i,j) in the array is the probability of making an observation type j in state i.

    Saves the matrix in env.observe_matrix and returns nothing.
    '''

    observe_probs = env.observe_probs
    state_types = env.state_types
    num_states = len(env.state_types)
    env.observe_matrix = np.zeros(shape=(num_states, env.num_observe_types))
    
    for i in range(num_states):
        env.observe_matrix[i] = observe_probs[state_types[i]]



def create_transition_matrices(env: Environment):
    '''
    If the transition_matrices in env is None, 
    constructs a 3D numpy array containing the transition matrix for each action.

    Entry (i,j,k) is the probability of transitioning from state j to k
    given that the agent takes action i.

    Saves the matrices in env.transition_matrices and returns nothing.
    '''

    if env.transition_matrices is not None: 
        # stachastic process (e.g. umbrella example)
        return
    else:
        # different actions cause different state transitions 
        # (e.g. robot example)
        
        act_eff = env.action_effects
        state_types = env.state_types
        
        i_len = len(act_eff)
        j_len = len(state_types)
        k_len = len(state_types)
        
        env.transition_matrices = np.zeros(shape=(i_len, j_len, k_len))
        
        for i in range(i_len):
            action = act_eff[i]
            for j in range(j_len):
                for offset in action.keys():
                    k = (j + offset) % j_len
                    env.transition_matrices[i][j][k] = action[offset]
        
        
        return
        

    
        
    
    
def normalize(l):
    sum = 0
    length = len(l)
    for i in l:
        sum += i
    for i in range(length):
        l[i] = l[i]/sum
        
def star_mult(x,y):
    '''
    x and y are both lists of the same length
    returns a list
    
    '''
    l = [0] * len(x)
    length = len(x)
    for i in range(length):
        l[i] = x[i]*y[i]
    
    return l



def forward_recursion(env: Environment, actions: List[int], observ: List[int], \
    probs_init: List[float]) -> np.ndarray:
    '''
    Perform the filtering task for all the time steps.

    Calculate and return the values f_{0:0} to f_{0,t-1} where t = len(observ).

    :param env: The environment.
    :param actions: The actions for time steps 0 to t - 2. Since we make an action to determine the NEXT state.
    :param observ: The observations for time steps 0 to t - 1.
    :param probs_init: The initial probabilities over the N states. That is, P(Si)
    :return: A numpy array with shape (t, N) (N is the number of states)
        the k'th row represents the normalized values of f_{0:k} (0 <= k <= t - 1).
    '''
    
    create_observation_matrix(env)
    create_transition_matrices(env)
    
    
    # each time step belongs to a state
    t = len(observ)
    N = len(env.state_types)
    
    all_f = np.zeros(shape = (t, N))
    
    # f{0:0}
    observe_type_zero = observ[0]
    sensor_model_zero = [0] * N
    transition_model_zero = probs_init
    
    for i in range(N):
        sensor_model_zero[i] = env.observe_matrix[i][observe_type_zero]
        
    f_zero_zero = star_mult(sensor_model_zero, transition_model_zero)
    normalize(f_zero_zero)
    all_f[0] = f_zero_zero
    
    # f{0:k}
    for i in range(1, t):
        action = actions[i-1]
        observe_type = observ[i]
        sensor_model = [0] * N
        last_f = all_f[i-1] # i.e. f{0:k-1}
        summation = [0] * N
        for j in range(N):
            transition_model = [0] * N
            # build the transition model
            for k in range(N):
                transition_model[k] = env.transition_matrices[action][j][k]*last_f[j]
            
            for s in range(N):
                summation[s] += transition_model[s]
                
        for t in range(N):
            sensor_model[t] = env.observe_matrix[t][observe_type]   
       
                    
        temp = star_mult(sensor_model, summation)
        normalize(temp)
        all_f[i] = temp
            
        
    
    
    
    return all_f



def backward_recursion(env: Environment, actions: List[int], observ: List[int] \
    ) -> np.ndarray:
    '''
    Perform the smoothing task for each time step.

    Calculate and return the values b_{1:t-1} to b_{t:t-1} where t = len(observ).

    :param env: The environment.
    :param actions: The actions for time steps 0 to t - 2.
    :param observ: The observations for time steps 0 to t - 1.

    :return: A numpy array with shape (t+1, N), (N is the number of states)
            the k'th row represents the normalized values of b_{k:t-1} (1 <= k <= t - 1),
            while the k=0 row is meaningless and we will NOT test it.
    '''
    
    create_observation_matrix(env)
    create_transition_matrices(env)

    
    t = len(observ)
    N = len(env.state_types)
    
    all_b = np.zeros(shape = (t+1, N))
    
    # Base case, b_{t:t-1}, all_b[t]
    
    all_b[t] = [1] * N
    
    # Recursion Case
    for k in range(t-2, -1, -1): # all_b[k+1], (1 <= k+1 <= t - 1, 0 <= k <= t - 2)
        
        summation = [0] * N
        action = actions[k]
        observe_type = observ[k+1]
        """
        print("k:", k)
        print("observation:", observe_type)
        print("action:", action)
        """
        for i in range(N): # for all possible values of s_{k+1}
            
            sensor_model = env.observe_matrix[i][observe_type] # constant
            
            next_b = all_b[k+2][i] # constant
            
            # print("sensor model:", sensor_model)
            # print("next_b:", next_b)
            
            transition_model = [0]*N # vector
            # print("transition model:")
            
            for j in range(N): # For all possible values of S_k
                # print(env.transition_matrices[action][j][i])
                transition_model[j] = sensor_model * next_b * env.transition_matrices[action][j][i]
            
            for s in range(N):
                summation[s] += transition_model[s]
        
        all_b[k+1] = summation
    
    return all_b


def fba(env: Environment, actions: List[int], observ: List[int], \
    probs_init: List[float]) -> np.ndarray:
    '''
    Execute the forward-backward algorithm. 

    Calculate and return a 2D numpy array with shape (t,N) where t = len(observ) and N is the number of states.
    The k'th row represents the smoothed probability distribution over all the states at time step k.

    :param env: The environment.
    :param actions: A list of agent's past actions.
    :param observ: A list of observations.
    :param probs_init: The agent's initial beliefs over states
    :return: A numpy array with shape (t, N)
        the k'th row represents the normalized smoothed probability distribution over all the states for time k.
    '''
    t = len(observ)
    N = len(env.state_types)
    fba = np.zeros(shape=(t, N))
    for i in range(t):
        f = forward_recursion(env, actions, observ, probs_init)[i]
        b = backward_recursion(env, actions, observ)[i+1]
        temp = star_mult(f, b)
        normalize(temp)
        fba[i] = temp
    return fba

