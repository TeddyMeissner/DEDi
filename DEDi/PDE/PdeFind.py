from typing import Union, List, Tuple

import jax
from jax import jit
from jax.experimental.sparse import BCOO

from functools import partial
import numpy as np
from scipy.sparse import lil_matrix

from DEDi.PDE.PDELib import *
from SparseJaxAD import *

from DEDi.DEDi import HeadDiffEqDiscovery

class PDE_find(HeadDiffEqDiscovery):
    
    def __init__(self,
                 observations: Union[np.ndarray, jnp.ndarray],
                 library:List[str],
                 TRange: Union[np.ndarray, jnp.ndarray, List],
                 XRange: Union[np.ndarray, jnp.ndarray, List],
                 YRange: Union[np.ndarray, jnp.ndarray, List] = None,
                 state_mask: Union[np.ndarray, jnp.ndarray, None] = None,
                 sparse_projections: Tuple[jnp.ndarray, jnp.ndarray, BCOO] = None,
                 setup_sparse = True,
                 smooth_init_states: Union[np.ndarray, jnp.ndarray] = None,
                 init_params: Union[np.ndarray, jnp.ndarray] = None,
                 cpu_or_gpu: str = 'cpu',
                 precision_64_bit: bool = True
                 ):
        """
        Looks to minimize ||u_t - f(u, theta)|| + lambda||u - u*|| + R(theta), where u* represents the data, and f is a library of candidate functions.
        
        Args:
            observations (Union[np.ndarray, jnp.ndarray]): Differential Equation data to be fit. Should be MxN where M is the number of time samples and N is spatial. 
            library (List[str]): List of candidate library functions as strings, e.g., ['u', 'u_x', 'u^2', 'u_x', ...].
            TRange (Union[np.ndarray, jnp.ndarray, List]): Array or list of initial and final time [T_0, T_f].
            XRange (Union[np.ndarray, jnp.ndarray, List]): Array or list of spatial endpoints [X_0, X_f].
            YRange (Union[np.ndarray, jnp.ndarray, List], optional): Array or list of secondary spatial endpoints [Y_0, Y_f]. Defaults to None.
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

        #Store inputs for later use
        self.observations = jnp.array(observations) 
        self.library = library
        self.PDEDimensions = self.observations.ndim - 1

        if self.PDEDimensions == 1:
            M,N_x = jnp.shape(observations)
            self.dy = None
        elif self.PDEDimensions == 2:
            M,N_x,N_y = jnp.shape(observations)
            self.dy = (YRange[1]-YRange[0])/(N_y-1) 

        #compute dt & dx for differentiation/integration
        self.dt = (TRange[1]-TRange[0])/(M-1)   
        self.dx = (XRange[1]-XRange[0])/(N_x-1) 

        if setup_sparse == True:
            if sparse_projections == None:
                print('Setting Up Sparse Structure')
                #Computes the max stencil in a sparse manner
                neighbor_x,neighbors_y = self.derivative_stencil_for_hess(self.library)
            
                self.sparsity,temp_color_vec = self.init_sparse_hessian(d_x=neighbor_x,d_y=neighbors_y) 
                self.projections = get_sparse_hessian_args(self.sparsity,coloring = "Star2",
                                                           colorvec=temp_color_vec)
            else:
                self.projections = sparse_projections

            print(f"Colors Needed = {np.shape(self.projections[0])[0]}") #Number of times to compute Hessian Vector Products
            self.SpHess = SparseHessian(self.objective,self.projections)
        else:
            self.SpHess = None

        ## Initialize instance of HeadDiffEqDiscovery
        super().__init__(observations,library,self.Model_Residual,state_mask = state_mask,
                         Sparse_Hessian = self.JITSparseHessian,ODE_or_PDE = 'PDE',
                         smooth_init_states=smooth_init_states, init_params = init_params)


    def compute_rhs(self,u,params):
        #Compute library function
        sz = jnp.shape(u)
        full_library = PDELibrary(u, self.library, self.dx, self.dy)
        rhs = jnp.reshape(full_library@params,sz)  
        return rhs

    def Model_Residual(self,u,params):  
        """ Computes the difference in u_{t+1} - integrate(u_{t}) using midpoint"""
        u_t = (u[1:] - u[0:-1])/self.dt
        u_mid =  self.compute_rhs((u[1:] + u[0:-1])/2,params)
        diff = u_t - u_mid
        
        #Disregard Boundaries
        if self.PDEDimensions == 1:
            return diff[:,2:-2]
        elif self.PDEDimensions == 2:
             return diff[:,2:-2,2:-2] 
    
    @partial(jit, static_argnums=(0,))
    def JITSparseHessian(self,x,param_mask,objective_params):
        """Function to compute and compile sparse Hessian"""
        return self.SpHess(x,param_mask,objective_params)


    def init_sparse_hessian(self, d_x=1, d_y=1):
        """Initializes a sparse Hessian matrix with specified spatial and temporal relationships."""
        paramsize = len(self.library)

        if self.observations.ndim == 3:
            numt,numx, numy = np.shape(self.observations)
            num_time = numx * numy
            num_vars = numt * numx * numy + paramsize

        elif self.observations.ndim == 2:
            numt,numx = np.shape(self.observations)
            num_time = numx 
            num_vars = numt * numx  + paramsize
            d_y = 0

        # Initialize diagonals and offsets collection
        diagonals = []
        offsets = []
        
        # Central and immediate temporal neighbors in x-direction
        if d_x !=0:
            #Neighbors in x direction
            for i in range(-2*d_x, 2*d_x + 1):
                offsets.extend([i, i + num_time, i - num_time])
                diagonals.extend([np.ones(num_vars) for _ in range(3)])

        if d_y!=0:
            # Neighbors in y-direction
            for j in range(-2*d_y, 2*d_y + 1):
                offsets_y_group = [j * numx, j * numx + num_time, j * numx - num_time]
                offsets.extend(offsets_y_group)
                diagonals.extend([np.ones(num_vars) for _ in offsets_y_group])

        if d_y !=0 and d_x != 0:
            #Overlap between stencil
            for i in range(-d_x, d_x + 1):
                offsets_x_group = [i - numx, i + numx, i - numx - num_time, i + numx - num_time, i - numx + num_time, i + numx + num_time]
                offsets.extend(offsets_x_group)
                diagonals.extend([np.ones(num_vars) for _ in offsets_x_group])
            for i in range(-d_y, d_y + 1):
                offsets_y_step_off = [i * numx - 1, i * numx + num_time - 1, i * numx - num_time - 1, i * numx + 1, i * numx + num_time + 1, i * numx - num_time + 1]
                offsets.extend(offsets_y_step_off)
                diagonals.extend([np.ones(num_vars) for _ in offsets_y_step_off])

        # Initialize the sparse matrix in 'lil' format and apply all changes
        hessian_sparse = lil_matrix((num_vars, num_vars))
        for diagonal, offset in zip(diagonals, offsets):
            hessian_sparse.setdiag(diagonal, offset)

        SparseStateStructure = BCOO.from_scipy_sparse(hessian_sparse)
        temp_color_vec = get_hess_color_vec(SparseStateStructure,coloring="Star2")
        maxcolor = jnp.max(temp_color_vec) + 1
        add_color = jnp.arange(maxcolor,maxcolor + paramsize)
        temp_color_vec = temp_color_vec.at[-paramsize:].set(add_color)

        hessian_sparse[-paramsize:, :] = 1
        hessian_sparse[:, -paramsize:] = 1

        SparseStructure = BCOO.from_scipy_sparse(hessian_sparse)
        
        return SparseStructure,temp_color_vec
    

    def derivative_stencil_for_hess(self,symbols):
        """Find max neighbors in stencil for centered finite difference"""

        def check_distance(x):
            if x == 0:
                return 0
            if x in [1,2]:
                return 1
            if x in [3,4]:
                return 2
            
        max_x_derivative, max_y_derivative = self.find_max_derivatives(symbols)
        
        return check_distance(max_x_derivative),check_distance(max_y_derivative)


    def find_max_derivatives(self,symbols):

        max_x_derivative = 0
        max_y_derivative = 0
        
        for symbol in symbols:
            max_x_derivative = max(max_x_derivative, symbol.count('x'),symbol.count('X'))
            max_y_derivative = max(max_y_derivative, symbol.count('y'),symbol.count('X'))

        return (max_x_derivative, max_y_derivative)
            
            