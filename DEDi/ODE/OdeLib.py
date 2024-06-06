from typing import Union, List, Callable
import numpy as np
import jax.numpy as jnp

def prepare_vectorized_lambda(X: Union[np.ndarray, jnp.ndarray], terms: List[str]) -> Callable:
    """Creates a callable function for the library terms from the size of the data and a list of the terms to be evaluated

    Args:
        X (Union[np.ndarray, jnp.ndarray]): Data to be passed to library function, needed for the shape of outputs
        terms (List[str]): String of terms in library i.e. ['x_1','x_2',....]

    Returns:
        Callable: function that takes in X and computes terms(X)
    """
    M,N = X.shape
    # Initialize the list for processed terms
    processed_terms = []
    terms = [term.replace('^', '**').replace(' ', '*') for term in terms]
    # Check if 'c' is in the terms and handle it separately
    if 'c' in terms:
        # Add a lambda part that generates a 2D column of ones with the correct shape
        processed_terms.append(f'jnp.ones((jnp.shape(x)[0], 1))')
    
    # Correctly map 'x_i' terms to 'x[:, i-1]' for array column access, ensuring they produce column vectors
    for term in terms:
        if term == 'c':
            continue  # Skip 'c' since it's already handled
        for i in range(1, N+1):  # Assuming you have less than 10 variables for simplicity
            if f'x_{i}' in term:
                term = term.replace(f'x_{i}', f'x[:, {i-1}:{i}]')  # Produce a 2D slice for each term
        processed_terms.append(term)

    # Construct the lambda function string with correct syntax
    # Use jnp.concatenate to combine the results along the second axis (columns), matching input orientation
    lambda_body = ",".join(processed_terms)
    lambda_str = f'lambda x: jnp.concatenate([{lambda_body}], axis=1)'

    # Use eval to create the lambda function from the string
    vectorized_func = eval(lambda_str, {'jnp': jnp})

    return vectorized_func

