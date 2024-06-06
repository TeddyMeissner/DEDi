import itertools
from typing import List

def PolyFunc(num_vars: int, degree: int, constant = False) -> List[str]:
    """Returns all polynomial terms in a deterministic order.

    Args:
        num_vars (int): Number of variables.
        degree (int): Degree of polynomial.
        constant (bool, optional): Whether a constant should be included. Defaults to False.

    Returns:
        List[str]: Polynomial terms.
    """
    variables = _generate_variable_names(num_vars)
    raw_terms = _generate_polynomial_terms(variables, degree)
    formatted_terms = _format_terms(raw_terms)
    formatted_terms.sort(key=_term_degree)
    if constant:
        formatted_terms.insert(0, 'c')
    return formatted_terms

def _generate_variable_names(num_vars: int) -> List[str]:
    return [f"x_{i}" for i in range(1, num_vars + 1)]

def _generate_polynomial_terms(variables: List[str], degree: int) -> List[tuple]:
    terms = []
    for d in range(1, degree + 1):
        for combination in itertools.combinations_with_replacement(variables, d):
            if combination not in terms:
                terms.append(combination)
    # Sorting to ensure determinism
    terms.sort()
    return terms

def _format_terms(terms: List[tuple]) -> List[str]:
    formatted_terms = []
    for term in terms:
        formatted_term = ''
        prev_var = ''
        count = 0
        for var in sorted(term):  # Sorting variables for consistent formatting
            if var == prev_var:
                count += 1
            else:
                if prev_var:
                    formatted_term += (prev_var if count == 1 else f"{prev_var}^{count}") + ' '
                prev_var = var
                count = 1
        formatted_term += (prev_var if count == 1 else f"{prev_var}^{count}")
        formatted_terms.append(formatted_term.strip())
    # Sorting the formatted terms to ensure they are in a consistent order before sorting by degree
    formatted_terms.sort()
    return formatted_terms

def _term_degree(term: str) -> int:
    degree = 0
    for part in term.split():
        if '^' in part:
            degree += int(part.split('^')[1])
        else:
            degree += 1
    return degree
