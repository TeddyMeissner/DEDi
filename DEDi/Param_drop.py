import jax.numpy as jnp
import numpy as np

def mask_lowest_params(params: np.ndarray, 
                       CurrentMask: np.ndarray, 
                       method: str = "single", 
                       iterations: int = 1) -> jnp.ndarray:
    
    """
    Computes a new mask based on the smallest parameters where the mask is non-zero.
    
    Args:
        params (np.ndarray): Parameters to be dropped.
        CurrentMask (np.ndarray): Defines which variables are available to drop.
        method (str, optional): Method to use for dropping parameters. Defaults to "single".
        iterations (int, optional): Number of times to use the defined method. For example, if method is 'single' and iterations is n, the n lowest parameters will be dropped. Defaults to 1.
    
    Returns:
        jnp.ndarray: New mask with 0's according to the smallest values in params.
    """

    ## Convert to numpy array for indexing
    params = np.array(params)

    def one_iter(params,CurrentMask,method):
        #Make sure parameters that are true zero, or tried twice are not used again
        params[np.where(CurrentMask == 0)] = np.inf
        #Make new param mask 
        CurrentMask = np.array(CurrentMask)

        if method == "bulk": 
            #Find the minimum in every column
            idx1 = np.argmin(abs(params),axis = 0)
            idx2 = np.arange(len(idx1))
            CurrentMask[(idx1,idx2)] = 0
        elif method == "single":
            #Find the overall minimum
            idx = np.unravel_index(np.argmin(abs(params),axis=None),params.shape)
            CurrentMask[idx] = 0
        else:
            print('Method Not Defined')
        return CurrentMask
    
    for _ in range(iterations):
        CurrentMask = one_iter(params,CurrentMask,method)

    return jnp.array(CurrentMask)

