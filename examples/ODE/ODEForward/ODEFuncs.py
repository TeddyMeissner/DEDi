from scipy.integrate import solve_ivp
import numpy as np

# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'RK45'
integrator_keywords['atol'] = 1e-12

def lorenz(t, x, sigma=10, rho=28, beta=8/3):
    return [
        sigma * (x[1] - x[0]),
        x[0] * (rho - x[2]) - x[1],
        x[0] * x[1] - beta * x[2],
    ]
    
def lorenzData(TSpan,dt,x0):
    #Tspan should be tuple
    T_grid = np.arange(TSpan[0],TSpan[1]+dt,dt)
    x_data_true = solve_ivp(lorenz, TSpan, x0, 
                    t_eval=T_grid, **integrator_keywords)
    
    return x_data_true.y.T

def vanderpol(t, x, mu=2.0):
    return [
        x[1],
        mu * (1 - x[0]**2) * x[1] - x[0]
    ]

def vanderpolData(TSpan, dt, x0):
    #Tspan should be tuple
    T_grid = np.arange(TSpan[0], TSpan[1] + dt, dt)
    x_data_true = solve_ivp(vanderpol, TSpan, x0, 
                            t_eval=T_grid, **integrator_keywords)
    
    return x_data_true.y.T

