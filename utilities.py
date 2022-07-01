import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def bs(S_0, K, r, sig, T):
    ''' 
    Returns Black Scholes call price for a given price of the underlying, strike, interest rate, vol., interest rate
    and time to maturity
    '''
    
    # define d1 and d2
    d1 = (np.log(S_0/K) + (r+0.5*sig**2)*(T)) / (sig*np.sqrt(T))
    d2 = d1 - sig*np.sqrt(T)
    
    # use BS pricing formula
    call_price = S_0*norm.cdf(d1) - K*np.exp(-r*(T))*norm.cdf(d2)
               
    return call_price

def bs_inv(P, S_0, K, r, T):
    '''
    Returns implied volatility from Black Scholes model for a given call price, strike, interest rate, vol., 
    interest rate and time to maturity
    '''
    
    # ensure at least intrinsic value
    P = np.maximum(P, np.maximum(S_0-K, 0))
    
    # loss function
    def diff(iv):
        return bs(S_0, K, r, iv, T) - P # simply return difference as the function does not work for squared difference for some reason?
    
    # minimze loss function -> find IV (circumvent ValueErrors)
    try:
        iv = brentq(diff, 1e-9, 100, maxiter=100000)
    except ValueError:
        iv = 0
        
    return iv

