from scipy.integrate import solve_ivp
import numpy as np

import numpy as np

def Report_Errors(theta_true: np.ndarray, theta_found: np.ndarray):
    """
    Compute true positives, false positives, and false negatives
    between two boolean masks.

    Args:
        theta_true (np.ndarray): Ground truth boolean array.
        theta_found (np.ndarray): Estimated boolean array.

    Returns:
        tuple: (TP, FP, FN)
    """
    #Relative L2
    L2_error = np.linalg.norm(theta_found - theta_true)/np.linalg.norm(theta_true)

    theta_true_bool,theta_found_bool = theta_true.astype(bool), theta_found.astype(bool)

    relative_inf_error = np.abs((theta_found - theta_true)[theta_true_bool] / theta_true[theta_true_bool])
    LInf_error = np.max(relative_inf_error)

    TP = np.sum(theta_true_bool & theta_found_bool)
    FP = np.sum(~theta_true_bool & theta_found_bool)
    FN = np.sum(theta_true_bool & ~theta_found_bool)
    true_positivity_ratio = TP/(TP + FN + FP)
    return L2_error, LInf_error, true_positivity_ratio


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
    T_grid = np.arange(TSpan[0],TSpan[1],dt)
    x_data_true = solve_ivp(lorenz, TSpan, x0, 
                    t_eval=T_grid, **integrator_keywords)
    
    return x_data_true.t,x_data_true.y.T

def getlorenzcoeffs(sigma=10, rho=28, beta=8/3):
    lorenz_coeffs = {
        'x_1': {'dot_x1': -sigma, 'dot_x2': rho},
        'x_2': {'dot_x1': sigma, 'dot_x2': -1},
        'x_3': {'dot_x3': -beta},
        'x_1*x_2': {'dot_x3': 1},
        'x_1*x_3': {'dot_x2': -1},
    }
    return lorenz_coeffs


def vanderpol(t, x, mu=2.0):
    return [
        x[1],
        mu * (1 - x[0]**2) * x[1] - x[0]
    ]

def vanderpolData(TSpan, dt, x0):
    #Tspan should be tuple
    T_grid = np.arange(TSpan[0], TSpan[1], dt)
    x_data_true = solve_ivp(vanderpol, TSpan, x0, 
                            t_eval=T_grid, **integrator_keywords)
    
    return x_data_true.t,x_data_true.y.T

def getvdpcoeffs(mu=2.0):
    vdp_coeffs = {
        'x_1': {'dot_x2': -1},
        'x_2': {'dot_x1': 1,'dot_x2': mu},
        'x_1**2*x_2': {'dot_x2': -mu},
    }
    return vdp_coeffs


def lorenz96(t, x, F = 8.0):
    N = len(x)
    dxdt = np.zeros(N)
    for i in range(N):
        dxdt[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
    return dxdt

def lorenz96Data(TSpan, dt, x0):
    #Tspan should be tuple
    T_grid = np.arange(TSpan[0], TSpan[1], dt)
    x_data_true = solve_ivp(lorenz96, TSpan, x0, 
                            t_eval=T_grid, **integrator_keywords)
    return x_data_true.t,x_data_true.y.T

######## LORENZ-96 ###########
def getlorenz96coeffs(F = 8.0):
    lorenz96_coeffs = {
        'x_2*x_5': {'dot_x1': 1},
        'x_4*x_5': {'dot_x1': -1},
        'x_1': {'dot_x1': -1},

        'x_1*x_3': {'dot_x2': 1},
        'x_1*x_5': {'dot_x2': -1},
        'x_2': {'dot_x2': -1},

        'x_2*x_4': {'dot_x3': 1},
        'x_1*x_2': {'dot_x3': -1},
        'x_3': {'dot_x3': -1},

        'x_3*x_5': {'dot_x4': 1},
        'x_2*x_3': {'dot_x4': -1},
        'x_4': {'dot_x4': -1},

        'x_1*x_4': {'dot_x5': 1},
        'x_3*x_4': {'dot_x5': -1},
        'x_5': {'dot_x5': -1},
        '1': {'dot_x1': F,'dot_x2': F,'dot_x3': F,'dot_x4': F,'dot_x5': F},
    }
    return lorenz96_coeffs


def colpitts(t,sol, a = 5, nu = 6.2723, gamma = .0797, q = .6898):
    x,y,z = sol
    dx = a*z
    dy = nu*(1 - np.exp(-x) + z)
    dz = -gamma*(x + y) - q*z
    return [dx, dy,dz]


def colpittsData(TSpan, dt, x0):
    T_grid = np.arange(TSpan[0], TSpan[1], dt)
    x_data_true = solve_ivp(colpitts, TSpan, x0, 
                            t_eval=T_grid, **integrator_keywords)
    return x_data_true.t,x_data_true.y.T

def getcolpittscoeffs(a = 5, nu = 6.2723, gamma = .0797, q = .6898):
    colpitts_coeffs = {
        'x_1': {'dot_x3': -gamma},
        'x_2': {'dot_x3': -gamma},
        'x_3': {'dot_x1': a,'dot_x2': nu,'dot_x3': -q},
        '1': {'dot_x2': nu},
        'exp(p*x_1)': {'dot_x2': -nu}
    }
    return colpitts_coeffs

def generate_coeff_matrix(functions, system_coeffs):
    num_variables = max(
        int(key.replace('dot_x', '')) for coeffs in system_coeffs.values() for key in coeffs.keys()
    )

    nopidx_system_coeffs = {
        delete_p_index(k): v for k, v in system_coeffs.items()
    }

    coeff_matrix = []
    for func in functions:
        func_nop = delete_p_index(func)
        coeffs = nopidx_system_coeffs.get(func_nop, {})
        coeff_row = [coeffs.get(f'dot_x{i+1}', 0) for i in range(num_variables)]
        coeff_matrix.append(coeff_row)

    return np.array(coeff_matrix)



import re

def delete_p_index(term):
    """
    Standardizes nonlinear terms like 'exp(p1*x_1)' to 'exp(p*x_1)'.
    """
    return re.sub(r'p\d+\*', 'p*', term)

