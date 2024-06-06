# Sparse regression method for discovery of ordinary and partial differential equations from incomplete and noisy data

This project implements a sparse regression strategy to discover ordinary and partial differential equations (ODEs and PDEs) from incomplete and noisy data. The inference is performed over both the equation parameters and state variables using a statistically motivated likelihood function. Sparsity is enforced by a selection algorithm that iteratively removes terms and compares models using statistical information criteria. Large-scale optimization is performed using a second-order variant of the Levenberg-Marquardt method, where the gradient and Hessian are computed via automatic differentiation. 

## Examples

Illustrations involving canonical systems of ordinary and partial differential equations are used to demonstrate the flexibility and robustness of the approach. Accurate reconstruction of systems is found to be possible even in extreme cases of limited data and large observation noise.

### 1. Ordinary Differential Equations (ODEs)
Lorenz equations:

$$
\begin{align*}
\dot{x} &= \sigma(y - x) \\
\dot{y} &= x(\rho - z) + y \\
\dot{z} &= xy - \beta y
\end{align*}
$$

$$(\sigma, \rho, \beta) = (10, 28, 8/3)$$

```python
# Full example code for ODE discovery found in examples/ODE/LorenzTest.ipynb where x_data_true comes from numerical solution

x_data_true = lorenzData(TSpan,dt,x0)
percent_noise = 0.5
noise = np.std(x_data_true,axis = 0)
x_data = x_data_true + np.random.normal(0,percent_noise*noise,size = x_data_true.shape)

Library = PolyFunc(num_vars=3,degree=2,constant=False)
discoverLorenz = ode(x_data,Library,[TSpan[0],TSpan[1]])

guess = discoverLorenz.FindBestModel(objective_params = [1e-2,1e-4],
                                     tolerance = 1e-10,verbose=True,
                                     max_iterations = 500,method = "LM",
                                     info_criteria='BIC',
                                     drop_method = {"initial":'single',"k":3},
                                     LA_solver = 'cholesky')

FullModels,latex_code = discoverLorenz.PrintModels(ICCutoff=10,return_latex = True)

opt_states,opt_params = discoverLorenz.unravel_states_and_parameters(discoverLorenz.states_and_parameters)

```

Found Equations:

$$\begin{align*}
\dot{x_1} &= -9.301x_1 + 9.370x_2 \\
\dot{x_2} &= 28.878x_1 - 1.137x_2 - 1.014x_1 x_3 \\
\dot{x_3} &= -2.660x_3 + 1.007x_1 x_2
\end{align*}$$

![Lorenz Image](examples/ODE/Images/Lorenz50pNoise.pdf)

### 2. Partial Differential Equations (PDEs)
Kuramoto-Sivashinsky Equation:
$$u_t = a uu_x + b u_{xx} +  c u_{xxxx}$$
$$(a,b,c)= (1,-1,-1)$$

```python
# Full example code for PDE discovery found in examples/PDE/KSTest.ipynb where u_total comes from numerical solution
percent_noise = 1.0
u_noise = u_total + np.random.normal(0,np.std(u_total)*percent_noise,size=u_total.shape)

symbols = ['u_xx','u_xxxx','u u_x','u','u^2','u^3','u_x','u_xxx','u^2 u_x']

trial = PDE(u_noise,symbols,TRange, XRange)

guess = trial.FindBestModel(objective_params = [1e-1,1e-4],
                                     tolerance = 1e-6,verbose=True,
                                     max_iterations = 500,method = "LM",
                                     info_criteria='BIC',
                                     drop_method = {"initial":'single',"k":3},
                                     LA_solver = "cholesky")

models,latex = trial.PrintModels(return_latex=True,ICCutoff=0)

opt_states,opt_params = trial.unravel_states_and_parameters(trial.states_and_parameters)

```
Found Equation:
$$u_t = -0.979u_{xx} - 0.991u_{xxxx} + 0.986u u_{x}$$

![KS Image](examples/PDE/Images/KS100pNoise0Missing.pdf)

## Software Requirements

This project requires the following software and libraries to be installed:

- **JAX**
- **scikit-sparse (sksparse)**
- **SciPy**



