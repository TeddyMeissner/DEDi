from typing import Union, List, Callable, Any
from functools import partial
import numpy as np
import scipy as sc

from jax import grad,jit,jvp
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from DEDi.optimizers import LM,SCMinimize
from DEDi.Param_drop import mask_lowest_params


class HeadDiffEqDiscovery:
    
    def __init__(self,
                 observations: Union[np.ndarray, jnp.ndarray],
                 library:List[str],
                 Model_Residual_func: Callable,
                 state_mask: Union[np.ndarray, jnp.ndarray, None] = None,
                 Sparse_Hessian: Callable = None,
                 ODE_or_PDE: str = "ODE",
                 smooth_init_states: Union[np.ndarray, jnp.ndarray] = None,
                 init_params: Union[np.ndarray, jnp.ndarray] = None
                 ):
        
        """
        Looks to minimize ||u_t - f(u, theta)|| + lambda||u - u*||, where u* represents the data, and f is a library of candidate functions.
        
        Args:
            observations (Union[np.ndarray, jnp.ndarray]): Differential Equation data to be fit. Should be MxN where M is the number of time samples and N is spatial. 
            library (List[str]): List of candidate library functions as strings, e.g., ['1', 'x', 'y', 'z', 'xy', 'xz', 'x^2'].
            Model_Residual_func (Callable): Function that returns the difference between the integrated points and the next time step, i.e., Model_Residual_func(u, parameters).
            state_mask (Union[np.ndarray, jnp.ndarray, None], optional): Array of 1s and 0s with the same shape as observations, where 0 indicates a gap in data. Defaults to None, meaning all data is usable.
            Sparse_Hessian (Callable, optional): Function to compute the sparse Hessian. If None, it will be computed densely and non-zero values will be extracted.
            ODE_or_PDE (str, optional): Specifies whether the problem is an "ODE" or "PDE". Defaults to "ODE".
            smooth_init_states (Union[np.ndarray, jnp.ndarray], optional): Initial smoothed states for optimization. Defaults to None.
            init_params (Union[np.ndarray, jnp.ndarray], optional): Initial parameters for optimization. Defaults to None.
        """
        
        #Store inputs for later use
        self.observations = jnp.array(observations) 
        self.library = library
        self.ODE_or_PDE = ODE_or_PDE
        self.integration = Model_Residual_func
        self.SparseHessian = Sparse_Hessian
        
        #Can Be changed after initilization if wanted
        self.SparsePenalty = "l0" 

        #Initial Parameter Guess & Parameter Mask
        if init_params is None:
            if ODE_or_PDE == 'ODE':
                _,N = jnp.shape(observations)
                parameters = 0.1*jnp.ones((len(library),N))
                self.param_mask = jnp.ones((len(library),N)).astype(bool)

            elif ODE_or_PDE == 'PDE': 
                parameters = [-1.0 if symbol == 'u_xxxx' else 1.0 for symbol in library]  #Well Posedness
                parameters = jnp.array(parameters).reshape(-1, 1)
                self.param_mask = jnp.ones((len(library),1)).astype(bool)
            else:
                raise ValueError('Method Not Defined')
        else:
            parameters = jnp.array(init_params)
            self.param_mask = jnp.ones_like(parameters).astype(bool)


        #Create State Mask for missing data. Make all ones if None
        if state_mask is None: 
            self.state_mask = jnp.ones_like(self.observations).astype(bool)
        else:
            self.state_mask = jnp.array(state_mask).astype(bool)

        #Store state, parameter, and data sizes for loss function normalization
        self.param_size = jnp.size(self.param_mask)
        self.state_size = jnp.size(self.observations)
        self.data_size = int(jnp.sum(self.state_mask.astype(bool).astype(int)))

        # If guess is provided for initialization use it, else linear interpolate if missing data
        if smooth_init_states is None: 
            if state_mask is not None: #If there is a state mask, fill in empty places with next time step
                initial_guess = self.fill_in_empty()
            else:
                initial_guess = self.observations
        else:
            initial_guess = jnp.array(smooth_init_states)

        #Initialize variables (u,theta) which are to be optimized, and function to bring them back to original shapes
        self.states_and_parameters,self.unravel_states_and_parameters = ravel_pytree((initial_guess,parameters))
        
        #Store Original (u,theta) for resetting optimization if wanted.  
        self.original_states_and_parameters = jnp.copy(self.states_and_parameters)
        
        #Initialize as nonzero number to test IC, if param_mask == 0: Stop. 3 here is arbitrary. 
        self.num_params_dropped = 3
        #Initalize CurrentLikelyhood and Mininimum LikelyHood for AIC to large arbitrary number. Will get overwrittern on first run. 
        self.CurrentLikely,self.MinLikely  = 1e10,1e10


    def fill_in_empty(self):

        def forward_interpolate(observations):
            x_tilde = jnp.pad(observations,pad_width=1,constant_values= .1)
            idx_row,idx_col = jnp.where(x_tilde == False) 

            while jnp.size(idx_row) > 0:
                x_tilde = x_tilde.at[idx_row,idx_col].set((x_tilde[idx_row+1,idx_col]))
                idx_row,idx_col = jnp.where(x_tilde == False)

            return x_tilde[1:-1,1:-1]

        if self.observations.ndim == 3:
            return jnp.array([forward_interpolate(sample) for sample in self.observations])
        else:
            return forward_interpolate(self.observations)


    def reset_variables(self):
        """Reset `states_and_parameters` to its original state."""
        self.states_and_parameters = self.original_states_and_parameters        


    def back_to_original_shape(self,variables,param_mask):
        """Reset `states_and_parameters` to its original shape and masks out dropped parameters."""
        observations,parameters = self.unravel_states_and_parameters(variables)
        parameters = param_mask*parameters
        return observations, parameters
    

    def sparsty_promotion_func(self,theta):
        """Penalized Parameters according to SparsePenalty, R(theta)"""
        if self.SparsePenalty == "l0":
            return 1 - jnp.e**(-theta**2/1)
        elif self.SparsePenalty == "l1":
            return jnp.abs(theta)
        elif self.SparsePenalty == "l2":
            return theta**2
        else:
            raise ValueError("Method Not Defined")


    def objective(self,states_and_parameters,param_mask,objective_params):
        """||u_t - N(u,theta)||_2^2 + \\lambda ||u - \\hat{u}||_2^2 + R(theta)"""

        lam,R = objective_params
        N,D,N_hat = self.state_size,self.param_size, self.data_size

        u,params = self.back_to_original_shape(states_and_parameters,param_mask)
        
        model_fit = (self.integration(u,params))
        observation_fit = ((u-self.observations)*self.state_mask) 

        f_1 = jnp.sum(model_fit**2)
        f_2 = jnp.sum((observation_fit)**2)*(N/N_hat)
        f_3 = jnp.sum(self.sparsty_promotion_func(params))/D
        self.f_1,self.f_2,self.f_3 = f_1,f_2,f_3
        
        return f_1 + lam*f_2 + R*f_3


    def LM_Minimize(self,*args,max_iterations = 400,tolerance = 1e-4,keep_track = False,verbose = True,
           **kwargs):
        """Levenberg-Marquardt optimization algorithm to minimize objective. See optimizers.LM """
        self.states_and_parameters = LM(self.states_and_parameters,self.JITObjective,self.JITGradient,self.SparseHessian,
                                    *args, max_iterations = max_iterations,tolerance = tolerance,
                                    keep_track = keep_track, verbose = verbose,**kwargs)
        return self.states_and_parameters
    

    def ScipyMinimize(self,*args,**kwargs):
        """Trust-ncg optimization algorithm to minimize objective. See optimizers.SCMinimize """
        def _jax_to_csc(A): #Takes in a Jax Sparse BCOO array and returns a Scipy CSC array 
            csc_mat = sc.sparse.coo_matrix((A.data, A.indices.T), shape=A.shape,dtype = float).tocsc()
            csc_mat.eliminate_zeros()
            return csc_mat
        
        if self.SparseHessian is None:
            hess = None
        else:
            def hess(x,*args):
                temp_hess = self.SparseHessian(x,*args)
                return _jax_to_csc(temp_hess)
            
        self.states_and_parameters = SCMinimize(self.JITObjective,self.states_and_parameters,
                              *args,jac = self.JITGradient,hessp = self.JITHVP, hess=hess,
                                **kwargs)
        
        return self.states_and_parameters
    

    def DropkLowest(self,params,method="bulk",iterations = 1):
        """Function to drop insignificant parameters and return updated mask. See Param_drop.mask_lowest_params"""
        self.param_mask = mask_lowest_params(params,self.param_mask,method=method,iterations=iterations)
        self.num_params_dropped = np.size(self.param_mask) - np.count_nonzero(self.param_mask)


    def FindBestModel(self, objective_params = [1e-2,1e-4], method = "LM",
                      max_iterations = 400, tolerance = 1e-4,verbose = False,info_criteria = 'BIC',**kwargs):
        """Function to find model which balances the best fit and fewest parameters using AIC or BIC. See Find_best_model.FindParsimoniousModel"""
        TestedModels = FindParsimoniousModel(self,objective_params,method,max_iterations,
                                             tolerance,verbose,info_criteria = info_criteria,**kwargs)
        self.tested_models = TestedModels


    def PrintModels(self,ICCutoff = None,verbose = True,return_latex = False):
        """Generates Latex for functions tested and orders them by AIC. Must use FindBestModel First"""
        return order_IC(self,self.tested_models,ICCutoff,verbose,return_latex=return_latex)


    def information_criteria(self,states_and_parameters ,param_mask,objective_params,method = 'BIC',tolerance = 0):
        """Computes AIC_c or BIC value for current model as defined by method"""

        N_hat,K = self.data_size, jnp.count_nonzero(param_mask)
        L_current = self.objective(states_and_parameters ,param_mask,objective_params)

        if method == 'AIC':
            info_current = N_hat*jnp.log(tolerance + L_current) + 2*K + 2*K*(K+1)/(N_hat-K-1)
        elif method == 'BIC':
            info_current = (N_hat)*jnp.log(tolerance + L_current) + np.log(N_hat)*K

        self.CurrentLikely = L_current

        if self.CurrentLikely < self.MinLikely:
            self.MinLikely = self.CurrentLikely

        return info_current,L_current

    #Jit Gradient, Hessians, and HVP for optimization
    @partial(jit, static_argnums=(0,)) 
    def JITObjective(self,x,param_mask,objective_params):
        """Function to compute and compile Objective"""
        return self.objective(x,param_mask,objective_params)


    @partial(jit, static_argnums=(0,)) 
    def JITGradient(self,x,param_mask,objective_params):
        """Function to compute and compile Gradient"""
        return grad(self.objective)(x,param_mask,objective_params)
    

    def HessVectorProd(self,fun):
        """Function to compute Hessian Vector Products"""
        def HVP(x,v,*args):
            return jvp(grad(lambda x: fun(x, *args)), (x,), (v,))[1]
        return HVP
    

    @partial(jit, static_argnums=(0,))
    def JITHVP(self,x,v,param_mask,objective_params):
        """Function to compute and compile Hessian Vector Products"""
        return self.HessVectorProd(self.objective)(x,v,param_mask,objective_params)
    
        
############# Function to optimize over everything
    
import threading
from queue import Queue
import copy 

np.set_printoptions(precision=3, suppress=True)

def FindParsimoniousModel(DiffEqClass:HeadDiffEqDiscovery,
                  objective_params: List[float],
                  optimization: str,
                  max_iterations: int = 500, 
                  tolerance: float = 1e-5,
                  verbose: bool = True,
                  info_criteria = 'BIC',
                  info_tol = 0,
                  drop_method = {"initial":'single',"k":3},
                  **kwargs : Any
):
    """
    Finds the best model by optimizing over the objective with and without the sparse penalty to find parameters to drop.
    Uses BIC/AIC to compare and accept or deny new models.
    
    Args:
        DiffEqClass (HeadDiffEqDiscovery): An instance of the HeadDiffEqDiscovery class.
        objective_params (List[float]): A list of floats for objective parameters.
        optimization (str): The method to use for optimizing (e.g., 'LM', 'Adam', 'trust-ncg').
        max_iterations (int, optional): The maximum number of iterations. Defaults to 500.
        tolerance (float, optional): The tolerance level for the optimization. Defaults to 1e-5.
        verbose (bool, optional): If True, print detailed information during the process. Defaults to True.
        info_criteria (str, optional): The information criteria to use for model comparison (e.g., 'BIC', 'AIC'). Defaults to 'BIC'.
        info_tol (float, optional): The tolerance level for information criteria. Defaults to 0.
        drop_method (Dict[str, Union[str, int]], optional): The method for dropping parameters. Defaults to {"initial": 'single', "k": 3}.
        **kwargs (Any): Additional keyword arguments to pass to the optimization procedure.
    """

    DroppingMethod, k = drop_method["initial"],drop_method["k"]

    def Minimize_ODE(DiffEqClass:HeadDiffEqDiscovery,objective_params):
        if optimization == "LM":
            return DiffEqClass.LM_Minimize(DiffEqClass.param_mask,objective_params, max_iterations=max_iterations,tolerance = tolerance,verbose = verbose,
                               **kwargs)
        else:
            return DiffEqClass.ScipyMinimize(DiffEqClass.param_mask,objective_params,maxiter=max_iterations,tol = tolerance,verbose = verbose,
                                      method = optimization,**kwargs)

    # Wrapper function to capture the return value of Minimize_ODE in a Queue
    def thread_target(queue, *args):
        result = Minimize_ODE(*args)
        queue.put(result)
    
    DiffEqClassL0_objective_params = np.copy(objective_params) #Include Sparse Penalty        
    DiffEqClass_objective_params = [objective_params[0],0.0] #No Sparse Penalty

    track_info_criteria = []

    currentICMin, adaptive_mode = np.inf, 0

    _,Best_Sparse_Params = DiffEqClass.back_to_original_shape(DiffEqClass.states_and_parameters,DiffEqClass.param_mask)
    Best_Sparse_Mask, CurrentBestStates_and_params = DiffEqClass.param_mask, DiffEqClass.states_and_parameters

    while DiffEqClass.num_params_dropped != 0:

        DiffEqClass.states_and_parameters = CurrentBestStates_and_params
        # Run Full and Sparse Model in Parrallel
        no_sparse_queue,sparse_queue = Queue(),Queue()
        p1 = threading.Thread(target=thread_target, args=(no_sparse_queue,DiffEqClass,DiffEqClass_objective_params))
        p2 = threading.Thread(target=thread_target, args=(sparse_queue,DiffEqClass,DiffEqClassL0_objective_params))
        p1.start()
        p2.start()
        p1.join()
        p2.join()
        no_sparse_result,sparse_result = no_sparse_queue.get(),sparse_queue.get()

        IC_proposed,L_current = DiffEqClass.information_criteria(no_sparse_result,DiffEqClass.param_mask,DiffEqClass_objective_params,method = info_criteria,tolerance = info_tol) # If want regularized set tolerance > 0        
        States_no_sparse,Params_no_sparse = DiffEqClass.back_to_original_shape(no_sparse_result,DiffEqClass.param_mask)
        _,Params_sparse = DiffEqClass.back_to_original_shape(sparse_result,DiffEqClass.param_mask)

        ChangeIC = (IC_proposed-currentICMin)

        if verbose:
            print(f'Function Balance: f_1 = {DiffEqClass.f_1:.4}, f_2 = {DiffEqClass.f_2:.4}, f_3 = {DiffEqClass.f_3:.4}')
            print("Proposed AIC",str(IC_proposed), ", Relative Change From Current Minimum ", str(ChangeIC))
            print("Direct Params", Params_no_sparse)
            print("L0 Params", Params_sparse,2)

        # Keep Track of the Equations attempted and their AIC value
        track_info_criteria.append([np.array(IC_proposed),np.array(L_current),np.array(Params_no_sparse),np.array(States_no_sparse),np.count_nonzero(Params_no_sparse)])

        if ChangeIC < 0:
            #Save Second best mask and params in case gets rejected again
            Second_Best_Sparse_Params,Second_Best_Sparse_Mask = Best_Sparse_Params , Best_Sparse_Mask
            #Save Current L0 params and Mask in case need to go back to them 
            Best_Sparse_Params,Best_Sparse_Mask = Params_sparse,copy.copy(DiffEqClass.param_mask)
            CurrentBestStates_and_params = sparse_result

            #Create Mask for next iteration, adaptive_mode = 1: dropping to less params, adaptive_mode = 2: first step forward was rejected, adaptive_mode = 3: Single step forward accepted
            if adaptive_mode == 0:
                DiffEqClass.DropkLowest(Best_Sparse_Params,method = DroppingMethod,iterations=k)
            elif adaptive_mode == 1 or adaptive_mode == 3:
                adaptive_mode = 3 #Means already accepted N parameters, and N-1 Parameters. No Need to move backwards. 
                DiffEqClass.DropkLowest(Best_Sparse_Params,method = DroppingMethod)
            elif adaptive_mode == 2: 
                NumToDrop = NumToDrop - 1
                DiffEqClass.param_mask = WorkBackwardMask
                DiffEqClass.DropkLowest(WorkBackwardParams,method = DroppingMethod,iterations=NumToDrop)
        else:
            if k != 1 and drop_method["initial"] == "bulk":
                k = k - 1
                DiffEqClass.DropkLowest(Best_Sparse_Params,method = DroppingMethod,iterations=k)
            else:
                if adaptive_mode == 0:
                    # Go Back to last time it was accepted
                    DiffEqClass.param_mask = Best_Sparse_Mask
                    #Drop a single parameter Instead 
                    DroppingMethod = "single"
                    DiffEqClass.DropkLowest(Best_Sparse_Params,method = DroppingMethod)
                    adaptive_mode = 1

                elif adaptive_mode == 1: 
                    #Go back 2 steps, N parameters was accepted, N - 1 rejected. Go to N + 1. 
                    WorkBackwardParams, WorkBackwardMask = Second_Best_Sparse_Params, Second_Best_Sparse_Mask
                    NumToDrop = np.count_nonzero(WorkBackwardMask) - np.count_nonzero(Best_Sparse_Mask) - 1
                    DiffEqClass.param_mask = WorkBackwardMask
                    DiffEqClass.DropkLowest(WorkBackwardParams,method = DroppingMethod,iterations=NumToDrop)
                    adaptive_mode = 2
                
                else:
                    break

        if verbose:
            print("Trying " + str(np.count_nonzero(DiffEqClass.param_mask)) + " Parameters")
  
        if IC_proposed < currentICMin:
            currentICMin = IC_proposed
        
        ## If going one by one no need to go back or forward
        if drop_method["initial"] == 'single' and drop_method["k"] == 1 and adaptive_mode == 1:
            break
    
    sorted_models = order_IC(DiffEqClass,track_info_criteria,None,verbose = False)
    opt_params,opt_states = sorted_models[0,2:4]
    DiffEqClass.states_and_parameters,_ = ravel_pytree((opt_states,opt_params))
    DiffEqClass.param_mask = opt_params.astype(bool)

    return sorted_models
        

############################################################################
def order_IC(DiffEqClass,track_info_criteria,ICCutoff,verbose, return_latex = False):
    import re
    from IPython.display import display, Latex
    '''This function is for ordering and printing all of the computed models'''
    
    def format_equation(coefficients, symbols, decimal_places=3):
        def format_symbol(s):
        # Replace underscore and subsequent characters (up to a space) with underscore and curly braces
            return re.sub(r"_(\w+)", r"_{\1}", s)
        
        if DiffEqClass.ODE_or_PDE == "ODE":
            terms = [f"{format(c, f'.{decimal_places}f')}{s}" for c, s in zip(coefficients, symbols) if c != 0]
        else:
            terms = [f"{format(c, f'.{decimal_places}f')}{format_symbol(s)}" for c, s in zip(coefficients, symbols) if c != 0]
        equation = " + ".join(terms).replace("+ -", "- ")
        return equation
    
    sorted_matrix = np.array(sorted(track_info_criteria, key=lambda row: row[0]),dtype=object)
    minAIC = sorted_matrix[0,0]
    sorted_matrix[:,0] = sorted_matrix[:,0] - minAIC
    sorted_matrix_for_plotting = sorted_matrix
    
    if ICCutoff != None:
        idxCutoff = np.where(sorted_matrix[:,0] <= ICCutoff)
        sorted_matrix = sorted_matrix[idxCutoff]
        
    M = np.shape(sorted_matrix)[0]
    
    latex_code = "\\begin{align*}"
    for j in range(M):
        latex_code += "Rel IC &= " + str(np.round(sorted_matrix[j,0],2)) + "\\\\"
        latex_code += "Likelyhood &= " + str(np.round(sorted_matrix[j,1],16)) + "\\\\"
        for i, row in enumerate(zip(*sorted_matrix[j,2])):
            equation = format_equation(row, DiffEqClass.library)
            if DiffEqClass.ODE_or_PDE == "ODE":
                latex_code += f"\\dot{{x_{i + 1}}} &= {equation} \\\\"
            else:
                latex_code += f"\\ u_t &= {equation} \\\\"
    latex_code += "\\end{align*}"

    if verbose:
        display(Latex(latex_code))

    if return_latex:
        return sorted_matrix_for_plotting,latex_code
    else:
        return sorted_matrix_for_plotting
