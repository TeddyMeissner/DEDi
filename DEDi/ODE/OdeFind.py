from typing import Union, List

import jax
from jax import jit

from functools import partial
import numpy as np

from DEDi.ODE.OdeLib import prepare_vectorized_lambda
from scipy.sparse import diags
from SparseJaxAD import *
from jax.experimental.sparse import BCOO


from DEDi.DEDi import HeadDiffEqDiscovery

class ode_find(HeadDiffEqDiscovery):
    
    def __init__(self,
                 observations: Union[np.ndarray, jnp.ndarray],
                 library:List[str],
                 TRange: Union[np.ndarray, jnp.ndarray, List],
                 state_mask: Union[np.ndarray, jnp.ndarray, None] = None,
                 sparse_projections: Tuple[jnp.ndarray, jnp.ndarray, BCOO] = None,
                 setup_sparse = True,
                 smooth_init_states: Union[np.ndarray, jnp.ndarray] = None,
                 init_params: Union[np.ndarray, jnp.ndarray] = None,
                 cpu_or_gpu: str = 'cpu',
                 precision_64_bit: bool = True
                 ):
        
        """
        Looks to minimize ||u_t - f(u, theta)|| + lambda||u - u*|| + R(theta), where u* represents the data, f is a library of candidate functions.
        
        Args:
            observations (Union[np.ndarray, jnp.ndarray]): Differential Equation data to be fit. Should be MxN where M is the number of time samples and N is spatial. 
            library (List[str]): List of candidate library functions as strings, e.g., ['1', 'x', 'y', 'z', 'xy', 'xz', 'x^2'].
            TRange (Union[np.ndarray, jnp.ndarray, List]): Array or list of initial and final time [T_0, T_f].
            state_mask (Union[np.ndarray, jnp.ndarray, None], optional): Array of 1s and 0s with the same shape as observations, where 0 indicates a gap in data. Defaults to None, meaning all data is usable.
            sparse_projections (Tuple[jnp.ndarray, jnp.ndarray, BCOO], optional): Precomputed projections from SparseJaxAD. If None, they will be computed.
            setup_sparse (bool, optional): Flag to setup sparse computations. Defaults to True.
            smooth_init_states (Union[np.ndarray, jnp.ndarray], optional): Initial smoothed states for optimization. Defaults to None.
            init_params (Union[np.ndarray, jnp.ndarray], optional): Initial parameters for optimization. Defaults to None.
            cpu_or_gpu (str, optional): Specifies whether to run computations on 'cpu' or 'gpu'. Defaults to 'cpu'.
            precision_64_bit (bool, optional): Flag to use 64-bit precision. Defaults to True.
        """
        
        jax.config.update('jax_platform_name', cpu_or_gpu)
        jax.config.update("jax_enable_x64", precision_64_bit)

        #Record original shape of observations
        self.observations = jnp.array(observations) 
        self.library = library
        M,_ = self.observations.shape

        #Compute dt for differentiation/integration - Could make this to allow for varying time steps
        self.dt = (TRange[1]-TRange[0])/(M-1)

        #Create function to compute the library terms 
        self.ODEVecLib = prepare_vectorized_lambda(self.observations,library)

        if setup_sparse == True:
            if sparse_projections == None:
                print("Setting up Sparse Hessian")
                #Computes the max stencil in a sparse manner
                self.sparsity,temp_color_vec = self.init_sparse_hessian() 
                self.projections = get_sparse_hessian_args(self.sparsity,coloring = "Star2",
                                                            colorvec=temp_color_vec)            
            else:
                self.projections = sparse_projections

            print(f"Colors Needed = {np.shape(self.projections[0])[0]}")  #Number of times to compute Hessian Vector Products
            self.SpHess = SparseHessian(self.objective,self.projections)
        else:
            self.SpHess = None

        ## Initialize instance of HeadDiffEqDiscovery
        super().__init__(observations,library,self.Model_Residual,state_mask = state_mask,
                         Sparse_Hessian = self.JITSparseHessian,ODE_or_PDE = 'ODE',
                         smooth_init_states=smooth_init_states, init_params = init_params)

    def Model_Residual(self,u,params):
        """ Computes the difference in u_{t+1} - integrate(u_{t}) using midpoint"""
        x_t = ( u[1:] - u[0:-1])/self.dt 
        library = self.ODEVecLib((u[0:-1] + u[1:])/2 )
        diff = x_t - library@params
        return diff


    @partial(jit, static_argnums=(0,))
    def JITSparseHessian(self,x,param_mask,objective_params):
        """Function to compute and compile sparse Hessian"""
        return self.SpHess(x,param_mask,objective_params)
        

    def init_sparse_hessian(self):
        """Initializes max sparse stencil for sparse hessian"""
        ## IMPLEMENTED AS MAX STENCIL
        NumVars = jnp.shape(self.observations)[1]
        paramsize = len(self.library)*NumVars

        MaxRelationships = NumVars*4 - 1 #4 works here for ODE Time before and after
        HessianSize = np.size(self.observations) + paramsize
        diagonals = [np.ones(HessianSize) for _ in range(MaxRelationships)]
        offsets = [i - int(MaxRelationships/2) for i in range(MaxRelationships)]
        
        sparsity = diags(diagonals, offsets, shape=(HessianSize, HessianSize), format= "lil")
        SparseStateStructure = BCOO.from_scipy_sparse(sparsity)
        temp_color_vec = get_hess_color_vec(SparseStateStructure,coloring="Star2")
        
        maxcolor = jnp.max(temp_color_vec) + 1
        add_color = jnp.arange(maxcolor,maxcolor + paramsize) #Do the Dense Parameters Separate from coloring
        temp_color_vec = temp_color_vec.at[-paramsize:].set(add_color)

        sparsity[-paramsize:,:] = 1
        sparsity[:,-paramsize:] = 1

        SparseStructure = BCOO.from_scipy_sparse(sparsity)

        return SparseStructure,temp_color_vec

