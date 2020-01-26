import numpy as np
from scipy.stats import norm

def get_df(t, ycrv_t, ycrv_rates, ycrv_freq = 9999):
    ''' 
        Discout factor function
        Discount factor for a horizon t: D(0,t) given we observe ycrv at time 0
        
        t         : time for which the discount factor is calculated 
        ycrv_t    : yield curve maturitities
        ycrv_rates: rates R(t) for maturities in ycrv_t
        ycrv_freq : coupounding frequency under which the rates are quoted. E.g. 2 means semiannual compounding
    '''
    return 1.0/(1.0+np.interp(t, ycrv_t, ycrv_rates)/ycrv_freq)**(t*ycrv_freq)

def fwd_swap_rate(T_start, T_end, swap_yearfrac, ycrv_t, ycrv_rates, ycrv_freq):
    ''' 
        function calculates forward swap rate R
        for a swap starting at T_start and ending at T_end
        
        T_start : start of the underlying swap
        T_end   : maturity of the underlying swap
        swap_yearfrac : coupon period of the underlying IRS, e.g. 0.5 for semiannually-paying swap
        ycrv_t  : yield curve maturitities
        ycrv_rates: rates R(t) for maturities in ycrv_t
        ycrv_freq: coupounding frequency under which the rates are quoted. E.g. 2 means semiannual compounding
    '''
    
    coupon_dates = np.arange(T_start, T_end+0.0001, swap_yearfrac)
    df = get_df(coupon_dates, ycrv_t, ycrv_rates, ycrv_freq)
    
    f = (df[0] - df[-1])/(swap_yearfrac * np.sum(df[1:]))
    return f

def swaption(T_start, T_end, K, IRS_type, sigma, swap_yearfrac, ycrv_t, ycrv_rates, ycrv_freq, model, shift = 0):
    '''
        Valuation of swaption at t=0. It is a function NOT depending on the 2-factor gaussian model, it is independent formula.
        The valuation is assuming either (shifted)lognormal (aka Black 76'), or normal model (where no shift applies)
        
        T_start : expiry of the swaption, and also start of the underlying swap
        T_end   : maturity of the underlying swap
        IRS_type: 'payer' or 'receiver'
        sigma   : volatility parameter in either (shifted) lognormal or normal model
        swap_yearfrac : coupon period of the underlying IRS, e.g. 0.5 for semiannually-paying swap
        ycrv_t  : yield curve maturitities
        ycrv_rates: rates R(t) for maturities in ycrv_t
        ycrv_freq: coupounding frequency under which the rates are quoted. E.g. 2 means semiannual compounding
        model: 'normal' or 'lognormal'
        shift: shift applicable to the lognormal model. Effectively all rates and strike will be bumped (up) by shift to prevent from negative values in log
    '''
    f            = fwd_swap_rate(T_start, T_end, swap_yearfrac, ycrv_t, ycrv_rates, ycrv_freq)
    coupon_dates = np.arange(T_start, T_end+0.0001, swap_yearfrac)
    df           = get_df(coupon_dates, ycrv_t, ycrv_rates, ycrv_freq)
    annuity      = swap_yearfrac*np.sum(df[1:])
    
    w = {'payer':-1, 'receiver':1}[IRS_type.lower()]
    
    if model.lower() == 'lognormal':
        d1      = (np.log((f+shift)/(K+shift)) + (0.5* sigma**2 * T_start))/(sigma*np.sqrt(T_start))
        d2      = d1 - sigma*np.sqrt(T_start)
        price   = (w*(K+shift)*norm.cdf(-w*d2) - w*(f+shift)*norm.cdf(-w*d1))*annuity
    elif model.lower() == 'normal':
        d       = -w * (f-K)*(sigma * np.sqrt(T_start))
        price   = sigma * np.sqrt(T_start) * (d * norm.cdf(d) + norm.pdf(d)) * annuity
    else:
        print(f'Swaption pricing: unknown model {model}')
      
    return price