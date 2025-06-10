import scipy as sc
from typing import Callable, Tuple, Any
import numpy as np
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from sksparse.cholmod import cholesky, CholmodError, analyze
import jax.scipy as jaxsc

############ Sparse Solvers
def sksparse_cholmod(
    H_csc: sc.sparse.csc_matrix,
    grad: jnp.ndarray,
    factor: Any,
    alpha_start: float,
    max_tries: int = 20,
    alpha_factor: float = 10.0,
) -> Tuple[Any, np.ndarray, float]:
    """
    Try to solve (H + alpha * I) x = -grad, increasing alpha if needed.
    Returns the used factor, direction, and how much alpha was added.
    """
    if factor is None:
        factor = analyze(H_csc)
    
    alpha = alpha_start
    # Save the unshifted diagonal
    original_diag = H_csc.diagonal().copy()
    for _ in range(max_tries):
        try:
            # Shift diagonal
            H_csc.setdiag(original_diag + alpha)

            factor.cholesky_inplace(H_csc)
            direction = factor.solve_A(grad)

            return factor, direction, alpha  # No extra shift needed

        except CholmodError:
            alpha *= alpha_factor

    raise RuntimeError(f"Failed to stabilize Hessian after {max_tries} tries (last alpha={alpha:.2e})")

## Iterative
def scipy_CG(A: BCOO, b: jnp.ndarray,maxiter = 100,**kwargs) -> np.ndarray: 
    """Solves Ax = b with sparse conjugate gradient and preconditioner incomplete LU decomposition using Scipy

    Args:
        A (BCOO)
        b (jnp.ndarray)

    Returns:
        np.ndarray:solution x
    """
    return sc.sparse.linalg.cg(A,np.array(b),maxiter=maxiter, atol=1e-10, rtol=1e-10)[0]


