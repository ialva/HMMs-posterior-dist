"""
Posterior Decoding for Hidden Markov Models

This module implements the forward-backward algorithm (posterior decoding)
for finding the most likely state at each position given an observation
sequence and an HMM.
"""

from typing import List
import numpy as np
from numpy.typing import NDArray


def posterior_decode(observation_seq: List[int], hmm) -> List[str]:
    """
    Decode an observation sequence using posterior (forward-backward) decoding.

    INPUT:
    - observation_seq: List of observations (integers indexing alphabet)
    - hmm: HMM object with states, initial_state_probs, transition_matrix,
      and emission_matrix

    OUTPUT:
    - state_seq: List containing the sequence of most likely states (state names)

    IMPLEMENTATION NOTES:
    1) The forward and backward matrices are implemented as matrices that are
       transposed relative to the way they are shown in class. Rows correspond
       to observations and columns to states.
    2) After computing the forward/backward probabilities for each observation,
       they are normalized by dividing by their sum. This maintains
       proportionality while avoiding numerical underflow.
    3) The posterior probability matrix is the element-wise product of the
       forward and backward matrices, normalized at each observation.
    """
    # YOUR CODE HERE
    posterior = posterior_probabilities(observation_seq,hmm)
    indices = np.nanargmax(posterior,axis=1)

    return [hmm.states[i] for i in indices]

def posterior_probabilities(observation_seq: List[int], hmm) -> NDArray[np.float64]:
    
    forward = _build_forward_matrix(observation_seq,hmm)
    backward = _build_backward_matrix(observation_seq,hmm)

    num_obs = len(observation_seq)
    num_states = hmm.num_states

    posterior = forward*backward
    row_sums = np.sum(posterior, axis=1, keepdims=True)
    posterior = np.divide(posterior, row_sums, out=np.zeros_like(posterior))

    return posterior

def _build_forward_matrix(observation_seq: List[int], hmm) -> NDArray[np.float64]:
    """
    Build the forward probability matrix.

    Similar to Viterbi but uses sum instead of max.
    Returns a matrix where rows are observations and columns are states.
    Each entry is normalized to avoid underflow.
    """
    # YOUR CODE HERE

    num_observations = len(observation_seq)
    num_states = hmm.num_states
    forward_matrix = np.zeros((num_observations,num_states))

    #first row of matrix is first obs. normalized values...

    first_obs = observation_seq[0]
    for x in range(num_states):
        
        if first_obs < len(hmm.emission_matrix):
            forward_matrix[0,x] = hmm.initial_state_probs[x] * hmm.emission_matrix[first_obs][x]
        else:
            forward_matrix[0,x] = 0.0
    
    norm_sum = np.sum(forward_matrix[0,:])
    if norm_sum > 0:
        forward_matrix[0,:] /= norm_sum
    else:
        
        forward_matrix[0,:] = 0.0
    
    
    for y in range(1,num_observations):
        obs = observation_seq[y]
        
        for j in range(num_states):
            total = 0.0
            for i in range(num_states):
                total+= forward_matrix[y-1,i] * hmm.transition_matrix[i][j]
            forward_matrix[y,j] = hmm.emission_matrix[obs][j] * total

        # Normalize to avoid underflow
        norm_sum = np.sum(forward_matrix[y,:])
        
        if norm_sum > 0:
            forward_matrix[y,:] /=  norm_sum
        else:
            # Handle impossible observation
            forward_matrix[y,:] = np.nan
 
    return forward_matrix


def _build_backward_matrix(observation_seq: List[int], hmm) -> NDArray[np.float64]:
    """
    Build the backward probability matrix.

    Works backwards from the last observation.
    Returns a matrix where rows are observations and columns are states.
    Each entry is normalized to avoid underflow.
    """
    # YOUR CODE HERE
    num_observations = len(observation_seq)
    num_states = hmm.num_states
    backward_matrix = np.zeros((num_observations,num_states))
    
    backward_matrix[-1,:] = 1.0 
    norm_sum = np.sum(backward_matrix[-1,:])
    backward_matrix[-1,:] /= norm_sum
    
    for y in range(num_observations-2,-1,-1):
        next_ob = observation_seq[y+1]
        for i in range(num_states):
            total = 0
            for j in range(num_states):
                total += hmm.transition_matrix[i][j] * hmm.emission_matrix[next_ob][j] * backward_matrix[y+1][j]
            backward_matrix[y,i] = total

        norm_sum = np.sum(backward_matrix[y,:])
        if norm_sum > 0:
            backward_matrix[y,:] /= norm_sum
        else:
            backward_matrix[y,:] = 0.0

    return backward_matrix


def _max_position(list_of_numbers: NDArray[np.float64]) -> int:
    """
    Find the index of the maximum value in a list.

    Returns the first index if there are ties or extremely close values.
    """
    max_value = -np.inf
    max_position = 0

    for i, value in enumerate(list_of_numbers):
        # This handles extremely close values that arise from numerical instability
        if value / max_value > 1 + 1E-5 if max_value > 0 else value > max_value:
            max_value = value
            max_position = i

    return max_position
