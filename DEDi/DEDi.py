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
                 RHS_Func: Callable,
                 state_mask: Union[np.ndarray, jnp.ndarray, None] = None,
                 validation_mask: Union[np.ndarray, jnp.ndarray, None] = None, 
                 Sparse_Hessian: Callable = None,
                 ODE_or_PDE: str = "ODE",
                 rescale_states = None,
                 smooth_init_states: Union[np.ndarray, jnp.ndarray] = None,
                 init_params: Union[np.ndarray, jnp.ndarray] = None,
                 NL_param_count = 0,
                 library_normalization = None
                 ):
        
        """
        Looks to minimize ||u_t - f(u, theta)|| + lambda||u - u*||, where u* represents the data, and f is a library of candidate functions.
        
        Args:
            observations (Union[np.ndarray, jnp.ndarray]): Differential Equation data to be fit. Should be MxN where M is the number of time samples and N is spatial. 
            library (List[str]): List of candidate library functions as strings, e.g., ['1', 'x', 'y', 'z', 'xy', 'xz', 'x^2'].
            Model_Residual_func (Callable): Function that returns the difference between the integrated points and the next time step, i.e., Model_Residual_func(u, parameters).
            RHS_Func (Callable): Function that computes the right-hand side f(x,p,t) of the differential equation model.
            state_mask (Union[np.ndarray, jnp.ndarray, None], optional): Array of 1s and 0s with the same shape as observations, where 0 indicates a gap in data. Defaults to None, meaning all data is usable.
            validation_mask (Union[np.ndarray, jnp.ndarray, None], optional): Optional mask to identify validation time points (for cross-validation in picking best hyperparameters).
            Sparse_Hessian (Callable, optional): Function to compute the sparse Hessian. If None, it will be computed densely and non-zero values will be extracted.
            ODE_or_PDE (str, optional): Specifies whether the problem is an "ODE" or "PDE". Defaults to "ODE".
            rescale_states (bool, optional): Whether to normalize each state dimension before fitting. Defaults to False.
            smooth_init_states (Union[np.ndarray, jnp.ndarray], optional): Initial smoothed states for optimization. Defaults to None.
            init_params (Union[np.ndarray, jnp.ndarray], optional): Initial parameters for optimization. Defaults to None.
            NL_param_count (int, optional): Number of nonlinear parameters (in addition to library coefficients).
            library_normalization (optional): Optional normalization vector for each library function. If None, no normalization is applied. Used to solve for true parameters. 
        """

        self.library = library
        self.ODE_or_PDE = ODE_or_PDE
        self.integration = Model_Residual_func
        self.RHS_func = RHS_Func
        self.SparseHessian = Sparse_Hessian
        self.rescale_states_vec = rescale_states

        #Can Be changed after initilization if wanted
        self.SparsePenalty = "l0" 

        self.unmasked_data = jnp.array(observations) #This is for validation error
        #Store inputs for later use
        if validation_mask is not None:
            if state_mask is not None:
                overlap = (~jnp.array(validation_mask)) & (~jnp.array(state_mask))
                if jnp.any(overlap):
                    print("[INFO] Validation mask includes entries with missing data. These will be ignored during validation error computation.")

                self.validation_mask = ~((~jnp.array(validation_mask)) & (jnp.array(state_mask)))
            else:
                self.validation_mask = jnp.array(validation_mask) 
        else:
            self.validation_mask = None

        self.observations = jnp.array(observations) if self.validation_mask is None else jnp.array(observations)*self.validation_mask 

        #Initial Parameter Guess & Parameter Mask
        if init_params is None:
            _,N = jnp.shape(observations)
            L_params = 1.0*jnp.ones((len(library),N))
            self.param_mask = jnp.ones((len(library),N)).astype(bool)
        else:
            L_params = jnp.array(init_params)
            self.param_mask = jnp.ones_like(L_params).astype(bool)

        NL_params = jnp.zeros(NL_param_count)

        #Create State Mask for missing data. Make all ones if None
        setup_state_mask = jnp.ones_like(self.observations).astype(bool) if state_mask is None else jnp.array(state_mask).astype(bool)
        #Hide validation states
        if validation_mask is not None:
            setup_state_mask *= self.validation_mask
        self.state_mask = setup_state_mask

        #Store state, parameter, and data sizes for loss function normalization
        self.param_size = jnp.size(self.param_mask)
        self.state_size = jnp.size(self.observations)
        self.data_size = jnp.sum(self.state_mask)

        # If guess is provided for initialization use it, else linear interpolate if missing data
        if smooth_init_states is None: 
            if state_mask is not None or validation_mask is not None: #If there is a state mask, fill in empty places with next time step
                initial_guess = self.fill_in_empty(self.observations)
            else:
                initial_guess = self.observations
        else:
            initial_guess = jnp.array(smooth_init_states)
        
        #Initialize variables (u,theta) which are to be optimized, and function to bring them back to original shapes
        self.states_and_parameters,self.unravel_states_and_parameters = ravel_pytree((initial_guess,L_params, NL_params))
        
        #Store Original (u,theta) for resetting optimization if wanted.  
        self.original_states_and_parameters = jnp.copy(self.states_and_parameters)
        
        #Initialize as nonzero number to test IC, if param_mask == 0: Stop. 3 here is arbitrary. 
        self.num_params_dropped = 3
        #Initalize CurrentLikelyhood and Mininimum LikelyHood for AIC to large arbitrary number. Will get overwrittern on first run. 
        self.CurrentLikely,self.MinLikely  = 1e10,1e10

        if library_normalization is None:
            self.normalize_lib = jnp.ones(len(library))
        else:
            self.normalize_lib = jnp.array(library_normalization)


    def fill_in_empty(self,observations):
        """Function to interpolate missing data for inital guess."""
        def interpolate_along_rows(obs):
            obs = np.array(obs, dtype=float)
            mask = obs != False  # True where we have real values

            x = np.arange(obs.shape[0])  # row index
            for col in range(obs.shape[1]):
                valid = mask[:, col]
                if np.sum(valid) == 0:
                    continue  # or raise warning
                obs[~valid, col] = np.interp(x[~valid], x[valid], obs[valid, col])
            return obs
        if observations.ndim == 3:
            return np.array([interpolate_along_rows(sample) for sample in observations])
        else:
            return interpolate_along_rows(observations)



    def reset_variables(self):
        """Reset `states_and_parameters` to its original state."""
        self.states_and_parameters = jnp.copy(self.original_states_and_parameters)   
        self.param_mask = jnp.ones_like(self.param_mask)
        #Initialize as nonzero number to test IC, if param_mask == 0: Stop. 3 here is arbitrary. 
        self.num_params_dropped = 3

    def back_to_original_shape(self,variables,param_mask, keep_normalized_flag = True):
        """Reset `states_and_parameters` to its original shape and masks out dropped parameters."""
        observations,L_params,NL_params = self.unravel_states_and_parameters(variables)
        parameters = param_mask*L_params
        if self.rescale_states_vec is None or keep_normalized_flag: 
           return observations, parameters, NL_params
        else: 
            return self.unscale_parameters(observations,L_params,NL_params)
    

    def sparsty_promotion_func(self,theta):
        """Penalized Parameters according to SparsePenalty, R(theta)"""
        if self.SparsePenalty == "l0":
            return 1 - jnp.e**(-theta**2/1.0)
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

        u,L_params, NL_params = self.back_to_original_shape(states_and_parameters,param_mask)
        model_fit = (self.integration(u,L_params,NL_params))
        observation_fit = ((u-self.observations)*self.state_mask)

        f_1 = jnp.sum(model_fit**2)/N
        f_2 = jnp.sum((observation_fit)**2)/N_hat
        f_3 = jnp.sum(self.sparsty_promotion_func(L_params))/D

        self.f_1,self.f_2,self.f_3 = f_1,f_2,f_3
        
        return N*(f_1 + lam*f_2) + R*(f_3)

    def LM_Minimize(self,*args,max_iterations = 500,tolerance = 1e-4,keep_track = False,verbose = True,
           **kwargs):
        """Levenberg-Marquardt optimization algorithm to minimize objective. See optimizers.LM """
        sol = LM(self.states_and_parameters,self.JITObjective,self.JITGradient,self.SparseHessian,
                                    *args, max_iterations = max_iterations,tolerance = tolerance,
                                    keep_track = keep_track, verbose = verbose,**kwargs)
        if keep_track:
            self.states_and_parameters = sol[1]
        else:
            self.states_and_parameters = sol
        return sol
    

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

    def DropkLowest(self,params,method="single",iterations = 1):
        """Function to drop insignificant parameters and return updated mask. See Param_drop.mask_lowest_params"""
        params = (params.T/self.normalize_lib).T
        self.param_mask = mask_lowest_params(params,self.param_mask,method=method,iterations=iterations)
        self.num_params_dropped = np.size(self.param_mask) - np.count_nonzero(self.param_mask)


    def FindBestModel(self, lambdas: List[float],Rs: List[float], method = "LM",
                      max_iterations = 500, tolerance = 1e-5,verbose = False,info_criteria = 'BIC',**kwargs):
        """Function to find model which balances the best fit and fewest parameters using AIC or BIC. See FindParsimoniousModel"""
        TestedModels = FindParsimoniousModel(self,lambdas,Rs,method,max_iterations,
                                             tolerance,verbose,info_criteria = info_criteria,**kwargs)
        self.tested_models = TestedModels

    def best_model(self):
        lam, r, BIC_val,validation_error, model_error, state_error, sparse_penalty, opt_states, opt_params, opt_NL_params = self.tested_models[0]
        return opt_params, opt_NL_params, opt_states

    def get_best_models_by_structure(self, tested_models):
        """
        Return indices of models with unique opt_param structures,
        keeping only the one with the lowest validation error per structure.
        """
        structure_map = {}

        for i, model in enumerate(tested_models):
            (_, _, _, validation_error, *_, opt_states, opt_params, opt_NL_params) = model
            # Ensure opt_params is 1D and convert to hashable tuple of bools
            bool_mask = tuple(np.ravel(np.array(opt_params)).astype(bool))

            if bool_mask not in structure_map:
                structure_map[bool_mask] = (i, validation_error)
            else:
                # Keep the one with the lower validation error
                if validation_error < structure_map[bool_mask][1]:
                    structure_map[bool_mask] = (i, validation_error)

        best_indices = [idx for idx, _ in structure_map.values()]
        return [tested_models[i] for i in best_indices]
    
    def PrintModels(self,N_best = 1,verbose = True,return_latex = False, tested_models = None):
        """Prints and generates Latex for functions tested assuming they are already ordered. Must use FindBestModel First"""
        tested_models = self.tested_models if tested_models is None else tested_models
        return print_models_sorted(tested_models, self.library, N_best=N_best, verbose=verbose, return_latex=return_latex)

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
    
    def unscale_parameters(self, observations,L_params,NL_params):
        Linear_param_flag = [("p" not in term) for term in self.library]
        Nonlinear_mask = np.array(Linear_param_flag).reshape(1, -1) 
        Nonlinear_scaling_idxs = self.get_nonlinear_p_scaling(self.library)
        NL_unscale = self.compute_scaling_factors(Nonlinear_scaling_idxs, 1/self.rescale_states_vec)

        scale = self.rescale_states_vec.copy()
        if scale.shape[0] != 1:
            scale = scale.reshape(1, -1)

        eval_func = self.RHS_func(1/scale, NL_params, self.t[0:1])
        eval_func = np.where(Nonlinear_mask, eval_func, 1.0)

        unscaled_L_params = (L_params*scale)*eval_func.T
        unscaled_NL_params = NL_unscale*NL_params
        unscaled_states = observations*self.rescale_states_vec

        return unscaled_states, unscaled_L_params, unscaled_NL_params

    def compute_scaling_factors(self, p_scaling_info, scale):
        factors = []
        for term in p_scaling_info:
            factor = 1.0
            for idx, power in term:
                factor *= scale[idx - 1] ** power  # idx is 1-based
            factors.append(factor)
        return np.array(factors)


    def get_nonlinear_p_scaling(self, library_terms):
        p_scaling = []

        for term in library_terms:
            p_match = re.search(r'p\d+', term)
            if p_match:
                pattern = r'p\d+\*([^\)]*)'
                expr_match = re.search(pattern, term)
                if expr_match:
                    expr = expr_match.group(1)
                    # Extract x components and powers
                    components = re.findall(r'x_(\d+)(?:\*\*(\d+))?', expr)
                    comp_info = [(int(i), int(p) if p else 1) for i, p in components]
                    p_scaling.append(comp_info)
                else:
                    # No matched x_i's so no scaling
                    p_scaling.append([])
        return p_scaling
    

############# Function to optimize over everything
import threading
from queue import Queue
import copy 

np.set_printoptions(precision=3, suppress=True)

def FindParsimoniousModel(DiffEqClass:HeadDiffEqDiscovery,
                  lambdas: List[float],
                  Rs: List[float],
                  optimization: str,
                  max_iterations: int = 500, 
                  tolerance: float = 1e-5,
                  verbose: bool = True,
                  info_criteria = 'BIC',
                  info_tol = 0,
                  output_sort = 'Validation',
                  drop_method = {"initial":'single',"k":5},
                  **kwargs : Any
):  
    all_solutions = []
    for lam in lambdas:
        for r in Rs:
            DiffEqClass.reset_variables()
            objective_params = [lam, r]
            _FindParsimoniousModel(DiffEqClass,objective_params,optimization, max_iterations, tolerance,
                                                   verbose,info_criteria, info_tol, drop_method,**kwargs)

            DiffEqClass.objective(DiffEqClass.states_and_parameters, DiffEqClass.param_mask, objective_params)
            _,state_error,sparse_penalty = DiffEqClass.f_1, DiffEqClass.f_2, DiffEqClass.f_3
            opt_states,opt_params,opt_NL_params = DiffEqClass.back_to_original_shape(DiffEqClass.states_and_parameters, DiffEqClass.param_mask,keep_normalized_flag=True)

            normalized_params = (opt_params.T*DiffEqClass.normalize_lib).T #Unnormalize for model error
            model_error = np.sum((DiffEqClass.integration(opt_states,normalized_params,opt_NL_params))**2)/np.size(opt_states)

            #Check Validation Error   
            if DiffEqClass.validation_mask is not None:
                N_val, K_val = np.sum(~DiffEqClass.validation_mask),np.count_nonzero(opt_params)
                validation = (opt_states - DiffEqClass.unmasked_data)*(~DiffEqClass.validation_mask)
                validation_error = np.sum(validation**2)/N_val
                BIC_val = (N_val)*jnp.log(validation_error) + jnp.log(N_val)*K_val

                val_string, bic_string =  f'{validation_error:.4e}', f'{BIC_val:.4e}'

            else:
                validation_error = "N/A"
                BIC_val = "N/A"
                val_string, bic_string = 'N/A', 'N/A'

            print(f"lam = {lam:.2e}, R = {r:.2e}, nonzero = {np.count_nonzero(opt_params)}, val_error = {val_string},BIC = {bic_string}, model_error = {model_error:.4e}, state_error = {state_error:.4e}")

            opt_states_forprint,opt_params_opt_states_forprint,opt_NL_params_forprint = DiffEqClass.back_to_original_shape(DiffEqClass.states_and_parameters, DiffEqClass.param_mask,keep_normalized_flag=False)
            all_solutions.append([lam, r, BIC_val, validation_error, model_error, state_error, sparse_penalty, 
                                  opt_states_forprint, opt_params_opt_states_forprint, opt_NL_params_forprint])

    # Sort solutions by validation error (ignoring "N/A" by treating them as infinity)
    if output_sort == "Validation":
        all_solutions.sort(key=lambda x: np.inf if x[3] == "N/A" else x[3])
    elif output_sort == "BIC":
        all_solutions.sort(key=lambda x: np.inf if x[2] == "N/A" else x[2])
    return all_solutions

def _FindParsimoniousModel(DiffEqClass:HeadDiffEqDiscovery,
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
    # DiffEqClass_objective_params = np.copy(objective_params) #No Sparse Penalty

    track_info_criteria = []

    currentICMin, adaptive_mode = np.inf, 0

    _, Best_Sparse_Params, _ = DiffEqClass.back_to_original_shape(DiffEqClass.states_and_parameters,DiffEqClass.param_mask, keep_normalized_flag = True)
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
        States_no_sparse,Params_no_sparse,Nonlinear_params = DiffEqClass.back_to_original_shape(no_sparse_result,DiffEqClass.param_mask, keep_normalized_flag=True)
        _,Params_sparse,_ = DiffEqClass.back_to_original_shape(sparse_result,DiffEqClass.param_mask, keep_normalized_flag = True)

        ChangeIC = (IC_proposed-currentICMin)

        if verbose:
            print(f'Function Balance: f_1 = {DiffEqClass.f_1:.4}, f_2 = {DiffEqClass.f_2:.4}, f_3 = {DiffEqClass.f_3:.4}')
            print("Proposed AIC",str(IC_proposed), ", Relative Change From Current Minimum ", str(ChangeIC))
            print("Direct Params", Params_no_sparse)
            print("L0 Params", Params_sparse,2)

        # Keep Track of the Equations attempted and their AIC value
        Params_normalized = (Params_no_sparse.T/DiffEqClass.normalize_lib).T
        track_info_criteria.append([np.array(IC_proposed),np.array(L_current),np.array(Params_normalized),np.array(Nonlinear_params),np.array(States_no_sparse),np.count_nonzero(Params_no_sparse)])

        if ChangeIC < 0:
            #Save Second best mask and params in case gets rejected again
            Second_Best_Sparse_Params, Second_Best_Sparse_Mask = Best_Sparse_Params, Best_Sparse_Mask
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

                if NumToDrop == 0:
                    break
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
    
    sorted_models = order_IC(track_info_criteria,None)
    opt_params,opt_NLparams,opt_states = sorted_models[0,2:5]
    DiffEqClass.states_and_parameters,_ = ravel_pytree((opt_states,opt_params,opt_NLparams))
    DiffEqClass.param_mask = opt_params.astype(bool)

    return sorted_models
        

############################################################################
def order_IC(track_info_criteria,ICCutoff):

    '''This function is for ordering and printing all of the computed models'''    

    sorted_matrix = np.array(sorted(track_info_criteria, key=lambda row: row[0]),dtype=object)
    minAIC = sorted_matrix[0,0]
    sorted_matrix[:,0] = sorted_matrix[:,0] - minAIC
    sorted_matrix_for_plotting = sorted_matrix
    
    if ICCutoff != None:
        idxCutoff = np.where(sorted_matrix[:,0] <= ICCutoff)
        sorted_matrix = sorted_matrix[idxCutoff]
        
    return sorted_matrix_for_plotting

### For Returning
import numpy as np
import re
from IPython.display import display, Latex

def format_equation(coefficients, symbols, decimal_places=3):
    def format_symbol(s):
        return re.sub(r"_(\w+)", r"_{\1}", s)
    
    terms = [f"{format(c, f'.{decimal_places}f')}{format_symbol(s)}" 
             for c, s in zip(coefficients, symbols) if c != 0]
    return " + ".join(terms).replace("+ -", "- ")

def substitute_nl_params(library, optNL_params, decimal_places=3):
    """Replace p1, p2, ... in library terms with values from optNL_params."""
    substituted_library = []
    param_index = 0
    for term in library:
        new_term = term
        for match in re.findall(r"p\d+", term):
            if param_index >= len(optNL_params):
                raise ValueError("optNL_params too short for library terms.")
            val = f"{optNL_params[param_index]:.{decimal_places}f}"
            new_term = new_term.replace(match, val)
            param_index += 1
        substituted_library.append(new_term)
    return substituted_library

def print_models_sorted(tested_models_sorted, library, N_best = 1, verbose=True, return_latex=False):
    """
    Print LaTeX for ODE models assuming tested_models_sorted is already ordered.

    Args:
        tested_models_sorted: List of lists [lam, r, BIC_val, Val_error, model_error, state_error, sparse_penalty, opt_states, opt_params, opt_NL_params]
        library: List of symbolic library terms
        ICCutoff: Max ΔIC allowed for printing
        verbose: Whether to print LaTeX
        return_latex: Return the latex string and filtered entries
    """
    
    min_BIC_index = np.argmin([model[2] for model in tested_models_sorted]) #Print \delta IC instead of IC
    min_BIC = tested_models_sorted[min_BIC_index][2] 
    
    filtered_models = tested_models_sorted[:N_best]

    latex_code = "\\begin{align*}\n"
    for model in filtered_models:
        lam, r, BIC_val,validation_error, model_error, state_error, _, _, opt_params, optNL_params = model

        if BIC_val == 'N/A':
            val_string, bic_string = 'N/A', 'N/A'
        else:
            delta_IC = BIC_val - min_BIC
            val_string, bic_string =  f'{validation_error:.4e}', f'{delta_IC:.4e}'


        latex_code += f"Validation~Error &= {val_string} \\\\\n"
        latex_code += f"\\Delta IC &= {bic_string} \\\\\n"
        latex_code += f"Model~Error &= {model_error:.3e} \\\\\n"
        latex_code += f"State~Error &= {state_error:.3e} \\\\\n"
        latex_code += f"\\lambda &= {lam:.2e}, \\quad R = {r:.2e} \\\\\n"

        updated_library = substitute_nl_params(library, optNL_params)
        for i, coeffs in enumerate(opt_params.T):
            eq = format_equation(coeffs, updated_library)
            latex_code += f"\\dot{{x}}_{i + 1} &= {eq} \\\\\n"

        latex_code += "\\\\\n"

    latex_code += "\\end{align*}"

    if verbose:
        display(Latex(latex_code))

    if return_latex:
        return filtered_models, latex_code
    else:
        return filtered_models
