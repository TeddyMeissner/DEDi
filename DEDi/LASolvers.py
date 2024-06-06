import scipy as sc
from typing import Tuple
import numpy as np
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from sksparse.cholmod import cholesky, CholmodError
import jax.scipy as jaxsc
from jax import jit

############## Sparse Solvers

## Direct
def sksparse_cholmod(A: BCOO, b: jnp.ndarray,Factor = None) -> Tuple[float, np.ndarray]:
    """Solves Ax = b with sparse cholesky decomposition. Shifts the diagonal until A is positive definite.

    Args:
        A (BCOO)
        b (jnp.ndarray)

    Returns:
        Tuple[float, np.ndarray]: Float to scale the diagonal, solution x
    """
    CSC_hess = _jax_to_csc(A)
    try: 
        if Factor is None:
            Factor = cholesky(CSC_hess)
        else: 
            Factor.cholesky_inplace(CSC_hess)
        return Factor,Factor.solve_A(b)
    
    except CholmodError:
        return Factor,jnp.zeros_like(b)

## Iterative
def scipy_CG(A: BCOO, b: jnp.ndarray,maxiter = 100) -> np.ndarray: 
    """Solves Ax = b with sparse conjugate gradient and preconditioner incomplete LU decomposition using Scipy

    Args:
        A (BCOO)
        b (jnp.ndarray)

    Returns:
        np.ndarray:solution x
    """
    CSC_hess = _jax_to_csc(A)
    return sc.sparse.linalg.cg(CSC_hess,np.array(b),maxiter=maxiter, atol=1e-10, rtol=1e-10)[0]

@jit
def jax_CG(A: BCOO, b: jnp.ndarray,M = None,maxiter = 100) -> np.ndarray: 
    """Solves Ax = b with sparse conjugate gradient and preconditioner incomplete LU decomposition using Scipy

    Args:
        A (BCOO)
        b (jnp.ndarray)

    Returns:
        np.ndarray:solution x
    """

    return jaxsc.sparse.linalg.cg(A.sort_indices(),b,M = M,maxiter=maxiter,tol=1e-10)[0]


#### Helper Functions
def _jax_to_csc(A)->sc.sparse.csc_matrix: 
    """Takes in a Jax Sparse BCOO array and returns a Scipy CSC array""" 
    csc_mat = sc.sparse.coo_matrix((A.data, A.indices.T), shape=A.shape,dtype = float).tocsc()
    csc_mat.eliminate_zeros()
    return csc_mat



