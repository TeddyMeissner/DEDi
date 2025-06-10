from typing import Union, List

import jax
from jax import jit

from functools import partial
import numpy as np

from DEDi.ODE.OdeLib import prepare_vectorized_lambda, count_nonlinear_params
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
                 validation_mask: Union[np.ndarray, jnp.ndarray, None] = None, 
                 sparse_projections: Tuple[jnp.ndarray, jnp.ndarray, BCOO] = None,
                 setup_sparse: bool = True,
                 rescale_states: bool = False,
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
            validation_mask (Union[np.ndarray, jnp.ndarray, None], optional): Optional mask to identify validation time points (for cross-validation in picking best hyperparameters).
            sparse_projections (Tuple[jnp.ndarray, jnp.ndarray, BCOO], optional): Precomputed projections from SparseJaxAD. If None, they will be computed.
            setup_sparse (bool, optional): Flag to setup sparse computations. Defaults to True.
            rescale_states (bool, optional): Whether to normalize each state dimension before fitting. Defaults to False.
            smooth_init_states (Union[np.ndarray, jnp.ndarray], optional): Initial smoothed states for optimization. Defaults to None.
            init_params (Union[np.ndarray, jnp.ndarray], optional): Initial parameters for optimization. Defaults to None.
            cpu_or_gpu (str, optional): Specifies whether to run computations on 'cpu' or 'gpu'. Defaults to 'cpu'.
            precision_64_bit (bool, optional): Flag to use 64-bit precision. Defaults to True.
        """
        
        jax.config.update('jax_platform_name', cpu_or_gpu)
        jax.config.update("jax_enable_x64", precision_64_bit)

        if rescale_states == True:
            if state_mask is not None:
                self.rescale_states = np.array([np.std(observations[state_mask[:, i], i]) for i in range(observations.shape[1])]) #Compute std over only known quantities
            else:
                self.rescale_states = np.std(observations, axis = 0)

            self.observations = jnp.array(observations/self.rescale_states) 
        else:
            self.rescale_states = None
            self.observations = jnp.array(observations) 

        #Record original shape of observations
        self.library = library
        M,N = self.observations.shape

        #Compute dt for differentiation/integration - Could make this to allow for varying time steps
        self.dt = (TRange[1]-TRange[0])/(M-1)
        self.t = jnp.linspace(TRange[0],TRange[1], M).reshape(-1,1)

        N_linear_params = N*len(library)
        N_nonlinear_params = count_nonlinear_params(library)
        total_params = N_linear_params + N_nonlinear_params

        #Create function to compute the library terms 
        self.ODEVecLib = prepare_vectorized_lambda(self.observations,library)

        if setup_sparse == True:
            if sparse_projections == None:
                print("Setting up Sparse Hessian")
                #Computes the max stencil in a sparse manner
                self.sparsity,temp_color_vec = self.init_sparse_hessian(N, total_params) 
                self.projections = get_sparse_hessian_args(self.sparsity,coloring = "Star2",
                                                            colorvec=temp_color_vec)            
            else:
                self.projections = sparse_projections

            print(f"Colors Needed = {np.shape(self.projections[0])[0]}")  #Number of times to compute Hessian Vector Products
            self.SpHess = SparseHessian(self.objective,self.projections)
        else:
            self.SpHess = None

        #Normalize library by normalizing inital library for stability
        init_lib = self.ODEVecLib(self.observations,np.zeros(N_nonlinear_params),self.t)
        normalize_lib_init = np.linalg.norm(init_lib, axis = 0)/np.sqrt(np.shape(init_lib)[0])
        normalize_lib = np.where(normalize_lib_init < 1e-5, 1, normalize_lib_init) #Numerical noise for constants

        ## Initialize instance of HeadDiffEqDiscovery
        super().__init__(self.observations,library,self.Model_Residual,self.ODEVecLib, 
                         state_mask = state_mask, validation_mask = validation_mask,
                         Sparse_Hessian = self.JITSparseHessian,ODE_or_PDE = 'ODE',
                         rescale_states = self.rescale_states,
                         smooth_init_states=smooth_init_states, init_params = init_params,
                         NL_param_count = N_nonlinear_params,
                         library_normalization = normalize_lib)
    
    def compute_rhs(self, u, p_L, p_NL, t):
        lib = self.ODEVecLib(u,p_NL, t)/self.normalize_lib
        return lib@p_L

    def Model_Residual(self,u,L_params, NL_Params):
        """ Computes the difference in u_{t+1} - integrate(u_{t}) using midpoint"""
        x_t = ( u[1:] - u[0:-1])/self.dt
        t_half =  self.t[1:]/2 + self.t[0:-1]/2
        rhs = self.compute_rhs((u[0:-1] + u[1:])/2,L_params, NL_Params, t_half)
        diff = x_t - rhs
        return diff


    @partial(jit, static_argnums=(0,))
    def JITSparseHessian(self,x,param_mask,objective_params):
        """Function to compute and compile sparse Hessian"""
        return self.SpHess(x,param_mask,objective_params)
        

    def init_sparse_hessian(self, num_vars, num_params):
        """Initializes max sparse stencil for sparse hessian"""
        ## IMPLEMENTED AS MAX STENCIL
        NumVars = num_vars
        paramsize = num_params

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

