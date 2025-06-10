import itertools
from typing import List, Optional
import re

def PolyFunc(num_vars: int,
             degree: int,
             constant: bool = False,
             custom: Optional[List[str]] = None) -> List[str]:
    """
    Generates polynomial terms up to a given degree in Python syntax.
    Optionally includes custom symbolic terms with unique parameter substitution.

    Args:
        num_vars (int): Number of variables (e.g., 3 creates (x_1, x_2, x_3)).
        degree (int): Max total polynomial degree.
        constant (bool): Include constant term ('1').
        custom (List[str], optional): List of custom symbolic strings using 'p' as a parameter placeholder.

    Returns:
        List[str]: List of symbolic strings like 'x_1**2', 'exp(p1*x_1)', etc.
    """
    variables = [f'x_{i+1}' for i in range(num_vars)]
    terms = []

    # Polynomial terms
    for d in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(variables, d):
            counts = {}
            for var in combo:
                counts[var] = counts.get(var, 0) + 1
            parts = [f'{var}**{power}' if power > 1 else var for var, power in sorted(counts.items())]
            terms.append('*'.join(parts))

    # Constant term
    if constant:
        terms.insert(0, '1')

    param_idx = 1
    if custom is not None:
        for expr in custom:
            def repl(match):
                nonlocal param_idx
                result = f'p{param_idx}'
                param_idx += 1
                return result

            # Replace whole words 'p' only
            new_expr = re.sub(r'\bp\b', repl, expr)
            terms.append(new_expr)

    return terms

