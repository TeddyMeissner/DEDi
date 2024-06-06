import jax.numpy as jnp

def PDELibrary(X, symbols, dx, dy):
    library = []
    for symbol in symbols:
        library.append(_GetPDEFunctions(X, symbol, dx, dy).flatten()) 
    return jnp.array(library).T


def _GetPDEFunctions(X, symbol,dx, dy):
    # Initialize the library of potential terms

    if symbol in ['u_y','u_yy','u_yyy','u_yyyy']: #Change axis of differentiation 
        step = dy
        axis = 0
    else:
        step = dx
        axis = 1

    if 'u' == symbol:
        def operator(d):
            return d
    elif 'u^2' == symbol:
        def operator(d):
            return d**2
    elif 'u^3' == symbol:
        def operator(d):
            return d**3
    elif 'u_x' == symbol or 'u_y' == symbol:
        def operator(d):
            return (jnp.roll(d, -1, axis=axis) - jnp.roll(d, 1, axis=1)) / (2*step)
    elif 'u_xx' == symbol or 'u_yy' == symbol:
        def operator(d):
            return (jnp.roll(d, -1,axis = axis) - 2 * d + jnp.roll(d, 1,axis = axis)) / step**2
    elif 'u_xxx' == symbol or 'u_yyy' == symbol:
        def operator(d):
            return (jnp.roll(d, -2,axis = axis) - 2 * jnp.roll(d, -1,axis = axis) + 2 * jnp.roll(d, 1,axis = axis) - jnp.roll(d, 2,axis = axis)) / (2 * step**3)
    elif 'u_xxxx' == symbol or 'u_yyyy' == symbol:
        def operator(d):
            return (jnp.roll(d, -2,axis = axis) - 4 * jnp.roll(d, -1,axis = axis) + 6 * d - 4 * jnp.roll(d, 1,axis = axis) + jnp.roll(d, 2,axis = axis)) / step**4
    elif 'u_xy' == symbol:
        return _GetPDEFunctions(_GetPDEFunctions(X,'u_x' ,dx,dy),'u_y',dx,dy)
    elif 'u_xxyy' == symbol:
        return _GetPDEFunctions(_GetPDEFunctions(X,'u_xx' ,dx,dy),'u_yy',dx,dy)
    else:
        parts = symbol.split()
        def operator(d):
            return _GetPDEFunctions(d,parts[0],dx,dy)*_GetPDEFunctions(d,parts[1],dx,dy)
    
    if X.ndim == 3:
        return jnp.array([operator(sample) for sample in X])
    else:
        return operator(X)