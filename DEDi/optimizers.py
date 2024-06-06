from typing import Callable, Tuple, Any
import numpy as np
import jax.experimental.sparse as jsparse
from DEDi.LASolvers import *
import time
from scipy.optimize import minimize
from jax.experimental.sparse import BCOO

def LM(x: jnp.ndarray, 
       f: Callable, 
       df: Callable, 
       dff: Callable, 
       *args: Tuple[Any, ...], 
       max_iterations: int = 200, 
       tolerance: float = 1e-4,
       verbose: bool = True, 
       keep_track: bool = False, 
       LA_solver: str = "cholesky", 
       max_LA_iters: int = 500,
       **kwargs: Any) -> jnp.ndarray:
    """ 
    Levenberg-Marquardt optimization algorithm.   

    Args:
        x (jnp.ndarray): Initial guess for the parameters.
        f (Callable): The objective function to be minimized.
        df (Callable): The first derivative of the objective function.
        dff (Callable): The second derivative (Hessian) of the objective function.
        args: Additional arguments to be passed to the objective function f.
        max_iterations (int, optional): Maximum number of iterations. Default is 200.
        tolerance (float, optional): Tolerance for convergence. Default is 1e-4.
        verbose (bool, optional): If True, print progress messages. Default is True.
        keep_track (bool, optional): If True, keep track of function values over iterations. Default is False.
        LA_solver (str, optional): Linear algebra solver to use. Default is "cholesky".
        kwargs: Additional keyword arguments to be passed to the objective function f.

    Returns:
        jnp.ndarray: Result of the optimization procedure
    """

    track = [] if keep_track else None
    cho_factor = None

    H = dff(x, *args, **kwargs)

    _FindDirection = FindDirection(method=LA_solver)
    
    alpha,iterations,LA_iterations = 100.0,0,30

    while True:
        iterations += 1
        grad = df(x, *args, **kwargs)

        if np.linalg.norm(grad) < tolerance:
            print("Converged, Gradient norm below tolerance")
            break

        # Compute Hessian and adjust with current alpha
        hessStart = time.time()
        H = dff(x, *args, **kwargs)
        hesstime = time.time() - hessStart        

        H_scaled = H + (alpha)*_getDiagonal(H,make_pos_def=True)

        # Find direction using a linear solver
        solvert = time.time()
        if LA_solver == 'cholesky':
            cho_factor,direction = _FindDirection(H_scaled,grad,Factor = cho_factor)
        else:
            direction = _FindDirection(H_scaled,grad,maxiter = LA_iterations)

        solvertime = time.time() - solvert

        solver_error = np.linalg.norm(H_scaled@direction - grad)

        f_old = f(x, *args, **kwargs)
        x_check = x-direction
        f_check = f(x_check, *args, **kwargs)

        # Update alpha based on the success of the step, and LA_iterations based on convergence
        if f_check < f_old:
            x = x_check
            if 1e-1 > solver_error/np.linalg.norm(grad): #Trust the step (Always satisfied with Cholesky), adjusts for iterative solver
                alpha /= 5
                LA_iterations /= 1.1
            else:
                LA_iterations *= 2
        else:
            alpha *= 3

        LA_iterations = int(np.min([LA_iterations,max_LA_iters]))
        
        new_norm = np.linalg.norm(df(x_check, *args, **kwargs))
    
        if verbose and np.mod(iterations,1) == 0:
            print(f"{iterations} | f = {f_check:.6f}, ||df|| = {np.linalg.norm(new_norm):.3e}, Alpha = {alpha:.3e}, SolverTime = {solvertime:.3f}s, HessianTime = {hesstime:.3f}s")
        if keep_track:
            track.append(x)

        #### Breaking conditions
        if iterations >= max_iterations:
            print("Max iterations reached, did not converge.")
            break
        
        if alpha < 1e-16:
            alpha = 1e-16
        if alpha > 1e16:
            print("Converged, function value is not changing")
            break
    
    if keep_track:
        return keep_track,x
    else:
        return x

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


### Helper Functions 

def _getDiagonal(H:jsparse.BCOO, make_pos_def = False)-> float:
    """
    Args:
        H (jsparse.BCOO): Sparse Hessian 
    Returns:
        float: max diagonal value, used to initialize alpha in LM 
    """
    # Access the data and indices
    data = H.data
    rows, cols = H.indices.T

    # Identify diagonal elements
    is_diagonal = rows == cols

    # Extract diagonal elements
    diagonal_data = data[is_diagonal]

    if make_pos_def: #Using Cholesky, diagonal should be positive
        diagonal_data = np.array(diagonal_data)
        diagonal_data[jnp.where(diagonal_data <= 0)] = 1e-8 

    return BCOO((diagonal_data,H.indices[is_diagonal]), shape = H.shape)


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
    
    elif method == 'jax_cg':
        return jax_CG

    else:
        raise ValueError("Linear Algebra Method Not Defined")