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
    # <snip>
    posterior_probs = _posterior_probabilities(observation_seq, hmm)
    numeric_state_seq = [_max_position(row) for row in posterior_probs]
    return [hmm.states[i] for i in numeric_state_seq]
    #</snip>

    # DEFINE _posterior_probabilites here
    #<snip>
def _posterior_probabilities(observation_seq: List[int], hmm) -> NDArray[np.float64]:
    """
    Calculate posterior state probabilities by combining forward and backward.

    Returns a matrix where rows are observations and columns are states.
    Each entry [i,j] is the posterior probability of state j at observation i.
    """
    # YOUR CODE HERE
    # <snip>
    forward = _build_forward_matrix(observation_seq, hmm)
    backward = _build_backward_matrix(observation_seq, hmm)

    # Element-wise multiplication
    posterior = forward * backward

    # Normalize each row to sum to 1.0
    row_sums = np.sum(posterior, axis=1, keepdims=True)
    posterior = posterior / row_sums

    return posterior
    # </snip>

def _build_forward_matrix(observation_seq: List[int], hmm) -> NDArray[np.float64]:
    """
    Build the forward probability matrix.

    Similar to Viterbi but uses sum instead of max.
    Returns a matrix where rows are observations and columns are states.
    Each entry is normalized to avoid underflow.
    """
    # YOUR CODE HERE
    # <snip>

    number_of_observations = len(observation_seq)
    number_of_states = hmm.num_states

    # Initialize forward matrix
    forward_matrix = np.zeros((number_of_observations, number_of_states))

    # First observation: initial state probs * emission probs
    forward_matrix[0] = (
        np.array(hmm.initial_state_probs) *
        np.array(hmm.emission_matrix)[observation_seq[0]]
    )
    # Normalize to avoid underflow
    total = np.sum(forward_matrix[0])
    if total > 0:
        forward_matrix[0] = forward_matrix[0] / total
    else:
        # Handle impossible observation
        forward_matrix[0] = np.nan

    # Fill in the rest of the matrix
    for observation_index in range(1, number_of_observations):
        # For each state, compute:
        # emission_prob * sum(prev_state_prob * transition_prob)
        emission_probs = np.array(
            hmm.emission_matrix)[observation_seq[observation_index]]

        # Compute all possible transitions from previous states
        # prev_probs * transition_matrix gives matrix where:
        # - rows are source states
        # - columns are destination states
        # We take the sum over source states for each destination
        prev_probs = forward_matrix[observation_index - 1]
        transition_matrix = np.array(hmm.transition_matrix)

        # Multiply each row of transition matrix by corresponding prev prob
        transitions = prev_probs[:, np.newaxis] * transition_matrix

        # Take sum over source states (rows) for each dest state (column)
        sum_transitions = np.sum(transitions, axis=0)

        # Multiply by emission probabilities
        forward_matrix[observation_index] = emission_probs * sum_transitions
  
        # </snip>
  
        # Normalize to avoid underflow
        total = np.sum(forward_matrix[observation_index])
        if total > 0:
            forward_matrix[observation_index] = forward_matrix[observation_index] / total
        else:
            # Handle impossible observation
            forward_matrix[observation_index] = np.nan
 
    return forward_matrix


def _build_backward_matrix(observation_seq: List[int], hmm) -> NDArray[np.float64]:
    """
    Build the backward probability matrix.

    Works backwards from the last observation.
    Returns a matrix where rows are observations and columns are states.
    Each entry is normalized to avoid underflow.
    """
    # YOUR CODE HERE
    # <snip>
    number_of_observations = len(observation_seq)
    number_of_states = hmm.num_states

    # Initialize backward matrix
    backward_matrix = np.zeros((number_of_observations, number_of_states))

    # Last observation: uniform distribution (all states equally likely to end)
    backward_matrix[number_of_observations - 1] = np.ones(number_of_states) / number_of_states

    # Fill in the rest of the matrix, working backwards
    for observation_index in range(number_of_observations - 2, -1, -1):
        # For each state at position observation_index, compute:
        # sum(transition_prob * backward[observation_index+1] * emission[observation_index+1])

        transition_matrix = np.array(hmm.transition_matrix)
        next_backward = backward_matrix[observation_index + 1]
        next_emission = np.array(hmm.emission_matrix)[observation_seq[observation_index + 1]]

        # For each source state, sum over all destination states:
        # sum_j (transition[i,j] * backward[observation_index+1, j] * emission[observation_index+1, j])
        for state_idx in range(number_of_states):
            backward_matrix[observation_index, state_idx] = np.sum(
                transition_matrix[state_idx, :] * next_backward * next_emission
            )

        # Normalize to avoid underflow
        total = np.sum(backward_matrix[observation_index])
        if total > 0:
            backward_matrix[observation_index] = backward_matrix[observation_index] / total
        else:
            # Handle edge case
            backward_matrix[observation_index] = np.nan
    #</snip>
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
