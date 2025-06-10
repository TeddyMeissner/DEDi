from typing import Union, List, Callable
import numpy as np
import jax.numpy as jnp
import re

def prepare_vectorized_lambda(X: Union[np.ndarray, np.ndarray], terms: List[str], return_param_count = False) -> Callable:
    """
    Creates a callable function f(x,p,t).

    Args:
        X (np.ndarray or jnp.ndarray): Input data (M x N)
        terms (List[str]): Symbolic library terms (e.g. ['x_1', 'sin(p1*x_1 + p2)', ...])

    Returns:
        Callable: A function (x, p, t)
    """
    M, N = X.shape
    processed_terms = []

    terms = [term.replace('^', '**')for term in terms]

    for term in terms:
        if term in ['1', 'c']:
            processed_terms.append('jnp.ones((x.shape[0], 1))')
            continue

        for i in range(1, N + 1):
            term = term.replace(f'x_{i}', f'x[:, {i-1}:{i}]')

        p_matches = sorted(set(re.findall(r'p\d+', term)), key=lambda s: int(s[1:]))
        nonlinear_term_count = 0
        for p_str in p_matches:
            idx = int(p_str[1:]) - 1
            term = term.replace(p_str, f'p[{idx}]')
            nonlinear_term_count += 1
        processed_terms.append(term)

    lambda_body = ', '.join(processed_terms)
    lambda_str = f'lambda x, p, t: jnp.concatenate([{lambda_body}], axis=1)'

    # Add trig and jnp to eval context
    eval_globals = {
        'jnp': jnp,
        'sin': jnp.sin,
        'cos': jnp.cos,
        'exp': jnp.exp,
        'tanh': jnp.tanh,
        'sqrt': jnp.sqrt,
        'log': jnp.log
    }
    
    return eval(lambda_str, eval_globals)


import re
from typing import List

def count_nonlinear_params(terms: List[str]) -> int:
    """
    Counts the number of unique nonlinear parameters p1, p2, ..., used in the symbolic terms.

    Args:
        terms (List[str]): List of symbolic expressions

    Returns:
        int: Number of unique nonlinear parameters (e.g., p1 → p8 → returns 8)
    """
    all_p_indices = set()
    for term in terms:
        matches = re.findall(r'p(\d+)', term)
        all_p_indices.update(int(idx) for idx in matches)
    return max(all_p_indices) if all_p_indices else 0


