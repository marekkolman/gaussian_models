import numpy as np
import pandas as pd
import basic_functions as bf
from scipy.optimize import fsolve
from scipy.stats import norm
from lmfit import Parameters
from tqdm import tqdm

class gaussian_model:
        
    def __init__(self, params, ycrv_t, ycrv_rates, ycrv_freq = 9999, integration_steps = 300):
        self.kappas, self.sigmas = self.__build_params_model(params)
        self.get_df              = lambda t: bf.get_df(t, ycrv_t, ycrv_rates, ycrv_freq)
        self.integration_steps   = integration_steps
        
    def __build_params_model(self, params):
        '''
            Build a 2-element vector 'kappas' and a 2x2 matrix 'sigmas' (expected by the model class) from a dictionary of parameters.
            This is necessary because there are many possible model parametrizations that result into different 'model parameters'
            Other parametrization could be possibly added here

            params: standard dictionary, or lmfit's 'dictionary' of parameters
        '''
        # params can be either lmfit's dict, or a standard dict, in either case it will be converted to a standard dict
        params = params.copy().valuesdict() if isinstance(params, Parameters) else params 

        if set(['kappa1', 'kappa2', 'sigma1', 'sigma2', 'rho']).issubset(params): # standard parametrization
            kappa1, kappa2, sigma1, sigma2, rho = params['kappa1'], params['kappa2'], params['sigma1'], params['sigma2'], params['rho']
            kappas = np.array([kappa1, kappa2])
            sigmas = np.array([[sigma1, sigma2*rho],[0, sigma2*np.sqrt(1-rho**2)]])

        if set(['kappa_r', 'kappa_e', 'sigma_r', 'sigma_e', 'rho']).issubset(params): #HW parametrization
            kappa_r, kappa_e, sigma_r, sigma_e, rho = params['kappa_r'], params['kappa_e'], params['sigma_r'], params['sigma_e'], params['rho']
            sigmas = kappa_r/(kappa_r-kappa_e)*np.array([[sigma_r-rho*sigma_e, -sigma_e*np.sqrt(1-rho**2)], 
                                                         [rho*sigma_e,         sigma_e*np.sqrt(1-rho**2)]])
            kappas = np.array([kappa_r, kappa_e])
        return kappas, sigmas
    
    def __g(self, t):
        '''
            Function g(t); which is 2x2 matrix composed of sigmas and kappas
        '''
        return self.sigmas*np.exp(t*self.kappas)

    def __h(self, t):
        '''
            Function h(t); which is a 2-element column vector only involving kappas
        '''
        return np.exp(-t*self.kappas)
    
    def __H(self, t):
        '''
            Function returns H(t)=diag(h(t)); the outcome is a 2x2 matrix, depending only on kappas
        '''
        return np.diag(self.__h(t))
    
    def __kappa_mat(self):
        '''
            Function transforms a vector of kappas to a diagonal matrix of kappas. Kappa_mat = diag(kappas)
        '''
        return np.diag(self.kappas)
    
    def __G(self, t, T):
        '''
            Function G(t, T, kappas). Mainly important for bond reconstitution formula. The outcome is a 2-element column vector
        '''
        return (1.0-np.exp(-(T-t)*self.kappas))/self.kappas
    
    def __y(self,t):
        '''
            4x4 matrix. Important for A(t,T) term in bond reconstitution formula P
        '''
        kappa1, kappa2 = self.kappas
        sigma11, sigma12, sigma21, sigma22 = self.sigmas.flatten()
        gg11 = (np.exp(2*kappa1*t)-1)*(sigma11**2 + sigma21**2)/(2*kappa1)
        gg12 = (np.exp(sum(self.kappas)*t)-1)*(sigma11*sigma12+sigma21*sigma22)/sum(self.kappas)
        gg21 = gg12
        gg22 = (np.exp(2*kappa2*t)-1)*(sigma12**2 + sigma22**2)/(2*kappa2)
        gg_integral = np.array([[gg11, gg12], [gg21, gg22]])
        return self.__H(t) @ gg_integral @ self.__H(t)
    
    def P(self, t, T, x):
        ''' 
            Function computes bond price given x = x(t): P(t, T| x(t) = x)
            It is a model bond reconstitution formula that builds bond price based on model parameters and observed yield curve
        '''
        Pt, PT = self.get_df(np.array([t, T]))
        A = -0.5* self.__G(t, T) @ self.__y(t) @ self.__G(t, T)
        return PT/Pt*np.exp(-self.__G(t, T) @ np.array(x) + A)
    
    def __var(self, T):
        '''
            Unconditional variances of x1(T), x2(T). The outcome is a vector of two values. Var[x1(T)], Var[x2(T)]
        '''
        sigma_square_sums = (np.transpose(self.sigmas)**2).sum(axis = 1)
        return (1-np.exp(-2*self.kappas*T))/(2*self.kappas)*sigma_square_sums
    
    def __cov(self, T):
        '''
            (Unconditional) covariance of x1(T), x2(T). The outcome is a vector of a single value Cov[x1(T),x2(T)]
        '''
        kappa_integral = (1-np.exp(-sum(self.kappas)*T))/sum(self.kappas)
        return sum(self.sigmas.prod(axis = 1))*kappa_integral
        
    def __mu1_cond(self, T, x2):
        '''
            Expected value of x1(T), conditional on x2(T) = x2
        '''
        return self.__cov(T)/self.__var(T)[1]*x2
    
    def __s1_sq_cond(self,T):
        '''
            Variance of x1(T), conditional on x2(T) = x2 
        '''
        var1, var2 = self.__var(T)
        return var1 - (self.__cov(T)**2)/var2
    
    
    def __swap_value(self, w, K, T0, TN, swap_yearfrac, x):
        '''
            Calcuation of a swap value (=payoff) at time T0 given 
            bond prices are model-based bond prices (P(T0,Ti) depend on x(T0))
            It is best to see this function as a in-model swap payoff
            
            w=-1 payer swap; w=1 receiver swap
        '''
        coupon_times = np.arange(T0+swap_yearfrac, TN+0.0001, swap_yearfrac)
        bond_prices = np.zeros_like(coupon_times)
        for i in range(len(coupon_times)):
            bond_prices[i] = self.P(T0, coupon_times[i], x)
        return -w*1 + w*bond_prices[-1] + w*K*swap_yearfrac*bond_prices.sum()

    def __get_strikes(self, w, K, T0, TN, swap_yearfrac, x2):
        '''
            For a given x2, find critical x1 = x1*(x2) and compute strikes Ki=P(T0,Ti, x1, x2) 
            for Jamshidian decomposition
        '''
        x1_crit = fsolve(lambda x1: self.__swap_value(w, K, T0, TN, swap_yearfrac, [x1, x2]), x0 = [0])[0]
        x = [x1_crit, x2]
        coupon_times = np.arange(T0+swap_yearfrac, TN+0.0001, swap_yearfrac)
        Ki = []
        for Ti in coupon_times:
            Ki.append(self.P(T0, Ti, x))
        return coupon_times, Ki
    
    def __bond_option(self, T, s, K, w, x2):
        '''
            2-factor gaussian model bond option value, conditional on x2(T)=x2. The option expires at T and is written on a s-bond
            
            T:  expiry of the option
            s:  maturity of the underlying bond
            w:  -1 = put, 1 = call
            x2: value of x2 upon which the bond option is conditional. It represents x2(T)=x2
        '''
        mu1       = self.__mu1_cond(T, x2)
        s1_sq     = self.__s1_sq_cond(T)
        G1, G2    = self.__G(T, s)
        omega     = -mu1*G1 + 0.5*(G1**2)*s1_sq
        A         = -0.5* self.__G(T, s) @ self.__y(T) @ self.__G(T, s)
        
        K_star    = self.get_df(T)/self.get_df(s)*np.exp(-A+x2*G2)*K
        d = lambda sign: (omega-np.log(K_star)+sign*0.5*(G1**2)*s1_sq)/(G1*np.sqrt(s1_sq))
        
        value = self.get_df(s)*np.exp(A-x2*G2)*(w*np.exp(omega)*norm.cdf(w*d(1))-w*K_star*norm.cdf(w*d(-1)))
        return value
    
    def swaption(self, K, T0, TN, swap_yearfrac, w):
        '''
            Full valuation formula for swaption at time t=0, knowing 2-factor gaussian model parameters
            The valuation formula involves numerical integration (over all value of x2) and is therefore slow
            
            K:              swaption strike
            T0:             expiry of the swaption and also the beginning of the underlying swap
            TN:             maturity of the underlying swap
            swap_yearfrac:  year fraction between two coupon dates in the IRS. For a swap that pays semiannually, use 0.5
            w:              -1 = payer swaption, 1 = receiver swaption
        '''
        
        def swaption_x2_cond(K, T0, TN, swap_yearfrac, w, x2):
            coupon_times, strikes = self.__get_strikes(-1, K, T0, TN, swap_yearfrac, x2)
            bond_option = []
            for Ti, Ki in zip(coupon_times, strikes):
                bond_option.append(self.__bond_option(T0, Ti, Ki, w, x2))    
            var_x2 = self.__var(T0)[1]
            return (bond_option[-1]+K*swap_yearfrac*sum(bond_option))/np.sqrt(var_x2)*norm.pdf(x2/np.sqrt(var_x2))
            
        x2_grid = np.linspace(-0.35, 0.35, self.integration_steps)
        swaption_val_x2 = []
        for x2 in x2_grid:
            swaption_val_x2.append(swaption_x2_cond(K, T0, TN, swap_yearfrac, w, x2))
        
        return np.trapz(swaption_val_x2, x2_grid)
    
    def __q_approx(self, t, T0, TN, swap_yearfrac, x = [0.0, 0.0]):
        '''
            Auxiliary function that is used in approximative swaption value
            x=[0.0,0.0] represents x(t):=[0.0,0.0] in integration. It is a reasonable assumption, therefore default values are set
        '''
        times = np.arange(T0, TN+0.0001, swap_yearfrac)
        bond_prices = []
        G_vals = []
        for T in times:
            bond_prices.append(self.P(t,T, x))
            G_vals.append(self.__G(t, T))
        bond_prices = np.array(bond_prices)
        G_vals = np.array(G_vals)
        G1_vals, G2_vals = G_vals[:,0], G_vals[:,1]
        
        A = swap_yearfrac * bond_prices[1:].sum()
        S = (bond_prices[0] - bond_prices[-1])/A
        
        q1 =(bond_prices[0]*G1_vals[0]-bond_prices[-1]*G1_vals[-1])/A-S*swap_yearfrac*(G1_vals[1:]*bond_prices[1:]).sum()/A
        q2 =(bond_prices[0]*G2_vals[0]-bond_prices[-1]*G2_vals[-1])/A-S*swap_yearfrac*(G2_vals[1:]*bond_prices[1:]).sum()/A
        
        return np.array([q1, q2])
    
    def swaption_approx(self, K, T0, TN, swap_yearfrac, w, x = [0.0, 0.0]):
        '''
            Swaption approximating formula in 2-factor gaussian model
            
            K:              swaption strike
            T0:             expiry of the swaption and also the beginning of the underlying swap
            TN:             maturity of the underlying swap
            swap_yearfrac:  year fraction between two coupon dates in the IRS. For a swap that pays semiannually, use 0.5
            w:              -1 = payer swaption, 1 = receiver swaption
            x:              optional value of x(t) set to a constant. [0.0, 0.0] is a reasonable compromise
        '''
        
        integrand = lambda t: np.linalg.norm(self.__q_approx(t, T0, TN, swap_yearfrac, x) @ np.transpose(self.sigmas))**2
        t_grid = np.linspace(0,T0, int(12*T0))
        integrand_val = []
        for t in t_grid:
            integrand_val.append(integrand(t))
        v = np.trapz(integrand_val, t_grid)
        
        times = np.arange(T0, TN+0.0001, swap_yearfrac)
        bond_prices = self.get_df(times)
        A = swap_yearfrac * bond_prices[1:].sum()
        S = (bond_prices[0]-bond_prices[-1])/A
        d = (S - K)/np.sqrt(v)
        return A*(w*(K-S)*norm.cdf(-w*d)+np.sqrt(v)*norm.pdf(w*d))

    
def get_err_lsq(params, df_vol_quotes, swap_yearfrac, ycrv_t, ycrv_rates, ycrv_freq):
    '''
        Objective (cost) function for fitting of 2-factor gaussian model to market ATM normal swaption volatilities. This function is supposed to be passed to lmfit's Minimize
        The cost measure are least squares are market vs model swaption values. This can be easily modified to squares of 'market vs model implied vols' to ensure equal weighting of each swaption
               
        params        : dictionary of input parameters, or lmfit's dictionary of parameters
        df_vol_quotes : dataframe with columns T0, TN, sigma_normal with market ATM quotes of normal volatilities on swaptions on T0 to TN swaps
        swap_yearfrac : coupon period of the underlying IRS, e.g. 0.5 for semiannually-paying swap
        ycrv_t        : yield curve maturitities
        ycrv_rates    : rates R(t) for maturities in ycrv_t
        ycrv_freq     : coupounding frequency under which the rates are quoted. E.g. 2 means semiannual compounding
    '''
    
    model = gaussian_model(params, ycrv_t, ycrv_rates, ycrv_freq)
    
    market_vals, model_vals = [], []
    for _, row in df_vol_quotes.iterrows():
        T0, TN, sigma_normal = row['T0'], row['TN'], row['sigma_normal']
        f = bf.fwd_swap_rate(T0, TN, swap_yearfrac, ycrv_t, ycrv_rates, ycrv_freq) 
        mkt_val   = bf.swaption(T0, TN, f, 'payer', sigma_normal, swap_yearfrac, ycrv_t, ycrv_rates, ycrv_freq, 'normal') #market value
        model_val = model.swaption_approx(f, T0, TN, swap_yearfrac, -1, x = [0.0, 0.0]) # model value
        market_vals.append(mkt_val)
        model_vals.append(model_val)
    sq_err = np.sqrt(((np.array(model_vals)-np.array(market_vals))**2).mean()) #just to display RMSE of the differences
    
    params = params.copy().valuesdict()
    if set(['kappa1', 'kappa2', 'sigma1', 'sigma2', 'rho']).issubset(params):
        print(f'kappa1: {params["kappa1"]:.6f}, kappa2: {params["kappa2"]:.6f}, sigma1: {params["sigma1"]:.6f}, sigma2: {params["sigma2"]:.6f}, rho: {params["rho"]:.6f}, error: {sq_err:.6f}')
    if set(['kappa_r', 'kappa_e', 'sigma_r', 'sigma_e', 'rho']).issubset(params):
        print(f'kappa_r: {params["kappa_r"]:.6f}, kappa_e: {params["kappa_e"]:.6f}, sigma_r: {params["sigma_r"]:.6f}, sigma_e: {params["sigma_e"]:.6f}, rho: {params["rho"]:.6f}, error: {sq_err:.6f}')
    return (np.array(model_vals)-np.array(market_vals))


def build_normal_volsurface(params, expiries, tenors, swap_yearfrac, ycrv_t, ycrv_rates, ycrv_freq):
    '''
        Given model parameters, build normal-volatility ATM volsurface that is represented by these gaussian two-factor model parameters
        The result is a dataframe with normal volatilities, in a pivot format
        
        params:   dictionary of input parameters, or lmfit's dictionary of parameters
        expiries: vector of expiries for the volsurface
        tenors:   vector of tenors for the volsurface
        swap_yearfrac : coupon period of the underlying IRS, e.g. 0.5 for semiannually-paying swap
        ycrv_t        : yield curve maturitities
        ycrv_rates    : rates R(t) for maturities in ycrv_t
        ycrv_freq     : coupounding frequency under which the rates are quoted. E.g. 2 means semiannual compounding
    '''
    expiries = pd.Series(expiries).astype(float)
    tenors   = pd.Series(tenors).astype(float)
    
    df_volsurface = pd.DataFrame(index=pd.MultiIndex.from_product([expiries, tenors], names = ['expiry', 'tenor'])).reset_index()
    model = gaussian_model(params, ycrv_t, ycrv_rates, ycrv_freq)
    
    normal_vols = []
    for _, row in tqdm(df_volsurface.iterrows(), total=len(df_volsurface)):
        T0, TN = row['expiry'], row['expiry']+row['tenor']
        K_ATM = bf.fwd_swap_rate(T0, TN, swap_yearfrac, ycrv_t, ycrv_rates, ycrv_freq)
        model_price = model.swaption_approx(K_ATM, T0, TN, swap_yearfrac, -1, x = [0.0, 0.0])
        sigma_normal = fsolve(lambda sigma: model_price - bf.swaption(T0, TN, K_ATM, 'payer', sigma, swap_yearfrac, ycrv_t, ycrv_rates, ycrv_freq, 'normal'), x0 = [0.005])[0]
        normal_vols.append(sigma_normal)
    df_volsurface['sigma'] = normal_vols
    df_volsurface = pd.pivot_table(df_volsurface, index = 'expiry', columns = 'tenor', values = 'sigma')
    return df_volsurface
