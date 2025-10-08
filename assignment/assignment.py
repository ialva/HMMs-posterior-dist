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

    # DEFINE _posterior_probabilites here

def _build_forward_matrix(observation_seq: List[int], hmm) -> NDArray[np.float64]:
    """
    Build the forward probability matrix.

    Similar to Viterbi but uses sum instead of max.
    Returns a matrix where rows are observations and columns are states.
    Each entry is normalized to avoid underflow.
    """
    # YOUR CODE HERE
  
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
