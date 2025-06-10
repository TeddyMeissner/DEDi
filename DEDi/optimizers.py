from typing import Callable, Tuple, Any
import numpy as np
import time
from DEDi.LASolvers import sksparse_cholmod,scipy_CG
from scipy.optimize import minimize
from jax.experimental.sparse import BCOO
import jax.numpy as jnp
import scipy as sc

def LM(x: jnp.ndarray,
       f: Callable,
       df: Callable,
       dff: Callable,
       *args,
       max_iterations: int = 200,
       tolerance: float = 1e-4,
       verbose: bool = True,
       keep_track: bool = False,
       LA_solver: str = "cholesky",
       max_LA_iters: int = 500,
       **kwargs) -> jnp.ndarray:
    """
    Sparse Levenberg-Marquardt optimization with fixed alpha strategy.
    """

    track = [] if keep_track else None
    N_x = x.size
    iterations = 0

    _FindDirection = FindDirection(method=LA_solver)

    # First call: initialize Hessian structure
    grad = df(x, *args, **kwargs)
    H_bcoo = dff(x, *args, **kwargs)

    H_csc = _jax_to_csc(H_bcoo)
    alpha = np.max(np.abs(H_csc.diagonal()))
    # Symbolic factorization once
    factor = None

    while True:
        if iterations >= max_iterations:
            print(f"[Failed to Converged] Iter {iterations}: ||grad|| = {grad_norm:.2e}")
            break
        iterations += 1

        # Gradient norm
        grad = df(x, *args, **kwargs)
        grad_norm = np.linalg.norm(grad) / np.sqrt(N_x)

        if grad_norm < tolerance:
            if verbose:
                print(f"[Converged] Iter {iterations}: ||grad|| = {grad_norm:.2e}")
            break

        # Update Hessian numerics
        hess_start = time.perf_counter()
        H_bcoo = dff(x, *args, **kwargs)
        hess_time = time.perf_counter() - hess_start

        H_csc = _jax_to_csc(H_bcoo)
        del H_bcoo

        # Solve
        solve_start = time.perf_counter()
        if LA_solver == "cholesky":
            factor, direction, alpha = _FindDirection(H_csc, grad, factor, alpha)
        else:
            H_csc.setdiag(H_csc.diagonal() + alpha)
            direction = _FindDirection(H_csc, grad, max_LA_iters)
        solve_time = time.perf_counter() - solve_start

        # Evaluate objective
        f_old = f(x, *args, **kwargs)
        x_trial = x - direction
        f_trial = f(x_trial, *args, **kwargs)

        #Revert Hess back for prediction: 
        H_csc.setdiag(H_csc.diagonal() - alpha)
        pred_decrease = -grad.dot(-direction) - 0.5 * direction.dot(H_csc @ (-direction))

        actual_decrease = f_old - f_trial
        rho = actual_decrease / pred_decrease if pred_decrease > 0 else -1.0

        # Accept or reject the step
        if actual_decrease > 0:
            x = x_trial
            if rho < 0.25:
                alpha = min(2*alpha, 1e16)
            elif rho > 0.75:
                alpha = max(.25*alpha,1e-16)
            else:
                pass
        else:
            alpha *= 2.0

        # Verbose
        if verbose:
            print(f"Iter {iterations:03d} | f={f_trial:.6e} | ||grad||={grad_norm:.2e} | Alpha={alpha:.2e} | Rho={rho:.2f}")
            print(f"  Hessian Time: {hess_time:.3f}s | Solve Time: {solve_time:.3f}s")
            print(f"x = {x[-20:]}")

        if keep_track:
            track.append([f(x, *args, **kwargs), grad_norm])

    return (track, x) if keep_track else x


def _jax_to_csc(A: BCOO) -> sc.sparse.csc_matrix:
    return sc.sparse.coo_matrix((A.data, A.indices.T), shape=A.shape).tocsc()



## Linear Agebra Helpers
def FindDirection(method = "cholesky")-> Callable:
    """Computes the solution to Hx = grad

    Args:
        H (jsparse.BCOO): Sparse Hessian
        grad (jnp.ndarray): gradient
        method (str, optional):  Defaults to "cholesky".

    Returns:
        np.ndarray: solution x
    """

    if method == 'cholesky':
        return sksparse_cholmod
    
    elif method == 'cg':
        return scipy_CG
    else:
        raise ValueError("Linear Algebra Method Not Defined")
    

def SCMinimize(fun:Callable,x:jnp.ndarray,*args: Tuple[Any, ...]
               ,method = 'trust-ncg',jac:Callable = None, hessp:Callable = None,
               hess:Callable = None,maxiter = 500,tol = 1e-5, verbose = True
               )-> np.ndarray:
    """
    Aims to minimize fun using defined method in scipy.optimize.minimize (Defaults to 'trust-ncg')

    Args:
        fun (Callable): The objective function to be minimized.
        x (jnp.ndarray): Initial guess for the parameters.
        args: Additional arguments to be passed to the objective function f.
        method (str, optional): Defaults to 'trust-ncg'.
        jac (Callable, optional): Function to compute the gradient.
        hessp (Callable, optional): Function to compute hessian vector product.
        maxiter (int, optional): Maximum number of iterations. Default is 500.
        tol (float, optional): Tolerance for convergence. Default is 1e-5.
        verbose (bool, optional): If True, print progress messages. Default is True.

    Returns:
        np.ndarray: solution form scipy trust-cg
    """
    return minimize(fun,x,args = args,method = method,jac = jac,
                    hessp = hessp,hess = hess,tol = tol,
                    options={'disp': verbose,'maxiter':maxiter,'gtol':tol}).x

