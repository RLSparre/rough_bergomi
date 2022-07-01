import numpy as np
from scipy.special import hyp2f1

class rBergomi:
    '''
    Cholesky Monte Carlo Pricer for the rough Bergomi model:
    '''
    
    
    def __init__(self, hurst, time_steps, time_to_maturity, rho, spot_variance, spot_price, vol_of_vol):
        
        self.H = hurst
        self.N = time_steps
        self.T = time_to_maturity
        self.rho = rho
        self.v0 = spot_variance
        self.S0 = spot_price
        self.nu = vol_of_vol
        self.chol_matrix = None
    
    
    # stores Cholesky decomposition of joint covariance matrix    
    def make_cholesky_matrix(self):
        ''' 
        Builds the joint covariance matrix of Z and B_hat (RLfBm) and then stores the Cholesky decomposition as an attribute
        of the class'''
        
        if self.chol_matrix is None:
        
            ### initialize variables
            
            # covariance matrix
            cov = np.zeros([2*self.N, 2*self.N])
            
            # time grid with N steps from dt to T
            t_grid = np.linspace(self.T/self.N, self.T, self.N) 
            
            # auxiliary variables
            gamma = 1/2 - self.H
            D_H = np.sqrt(2*self.H)/(self.H+1/2)
            
            ### define auxiliary functions
            
            # function G(x) (hyp2f1 is the Gauss hypergeometric function)
            def G(x):
                return 2*self.H*(
                    1/(1-gamma)*x**(-gamma)+gamma/(1-gamma) * x ** (-1-gamma) * 1/(2-gamma) * hyp2f1(
                        1, 1+gamma, 3-gamma,1/x
                    )
                )
            
            # translate index to time
            def itt(x):
                return t_grid[x % self.N]
            
            ### build covariance matrix for (Z, B_hat)
            for i, j in np.ndindex(cov.shape):
                if i < self.N and j < self.N:
                    cov[i,j] = min(itt(i), itt(j))
                    
                elif i >= self.N and j >= self.N:
                    if i == j:
                        cov[i,j] = itt(i)**(2*self.H) # variance of B_hat
                        
                    else:
                        max_time = max(itt(i), itt(j))
                        min_time = min(itt(i), itt(j))
                        cov[i,j] = min_time**(2*self.H)*G(max_time/min_time) # covariance B_hat_s (max_time), B_hat_t (min_time) 
                
                elif i < self.N and j >= self.N:
                    cov[i,j] = self.rho*D_H*(
                        itt(j)**(self.H+1/2)-(itt(j)-min(itt(i), itt(j)))**(self.H+1/2) # covariance B_hat_s and Z_t
                    ) 
                    
                elif i >= self.N and j < self.N:
                    cov[i,j] = self.rho*D_H*(
                        itt(i)**(self.H+1/2)-(itt(i)-min(itt(i), itt(j)))**(self.H+1/2) # covariance B_hat_s and Z_t
                    ) 
                    
            # store Cholesky decomposition as an attribute of the class
            self.chol_matrix = np.linalg.cholesky(cov)
            
            
    def get_joint_paths(self, number_of_paths):
        ''' 
        Returns required number of paths of joint process (Z, B_hat) starting from spot
        '''
        
        M = number_of_paths
        
        # dot product of Cholesky decomposition M paths of length 2N
        data = np.dot(self.chol_matrix, np.random.randn(self.chol_matrix.shape[1], M))
        
        # Brownian motion paths
        bm_paths = np.zeros([self.N+1, M])
        bm_paths[1:, :] = data[:self.N, :] # first N paths are Brownian motion
        bm_paths = bm_paths.transpose()
        
        # fractional Brownian motion paths
        fbm_paths = np.zeros((self.N+1, M))
        fbm_paths[1:, :] = data[self.N:, :]
        fbm_paths = fbm_paths.transpose()

        return bm_paths, fbm_paths
    
    
    def get_price_paths(self, number_of_paths):
        '''
        Returns required number of asset price paths
        '''
        
        
        # initialize
        M = number_of_paths
        price_paths = np.zeros([M, self.N+1])
        t_grid = np.linspace(0, self.T, self.N+1) 
        
        # get paths
        bm_paths, fbm_paths = self.get_joint_paths(M)
        
        # create vol. and price paths
        #vol_paths = np.sqrt(self.v0 * np.exp(self.nu*fbm_paths))
        vol_paths = np.sqrt(self.v0 * np.exp(self.nu*fbm_paths - 0.5*self.nu**2*t_grid**(2*self.H)))
        
        # initialize with S0
        price_paths[:, [0]] = self.S0 * np.ones((M, 1))
        
        S = price_paths[:, [0]]
        
        # fill price paths
        for j in range(1, self.N+1):
            S += S*vol_paths[:, [j-1]] * (bm_paths[:, [j]] - bm_paths[:, [j-1]])
            
            price_paths[:, [j]] = S
            
        return price_paths, vol_paths
    

    
    def get_log_price_paths(self, number_of_paths):
        '''
        Returns required number of log stock prices
        '''
        
        # initialize
        M = number_of_paths
        t_grid = np.linspace(0, self.T, self.N+1) 
        dt = self.T/self.N
        log_price_paths = np.zeros((M, self.N + 1))
        
        # get paths
        bm_paths, fbm_paths = self.get_joint_paths(M)
        
        # create vol. and price paths
        #vol_paths = np.sqrt(self.v0 * np.exp(self.nu*fbm_paths))
        vol_paths = np.sqrt(self.v0 * np.exp(self.nu*fbm_paths - 0.5*self.nu**2*t_grid**(2*self.H)))
        
        log_price_paths[:, [0]] = np.log(self.S0) * np.ones((M, 1))

        S = log_price_paths[:, [0]]

        for j in range(1, self.N + 1):

            S += -0.5 * vol_paths[:, [j-1]]**2 * dt +  \
                 vol_paths[:, [j-1]] * (bm_paths[:, [j]] - bm_paths[:, [j-1]])

            log_price_paths[:, [j]] = S

        return log_price_paths, vol_paths
    
    
    def get_log_prices(self, number_of_prices):
        """
        Returns number of log stock prices.
        """

        M = number_of_prices
        t_grid = np.linspace(0, self.T, self.N+1) 
        dt = self.T/self.N

        bm_paths, fbm_paths = self.get_joint_paths(M)

        vol_paths = np.sqrt(self.v0 * np.exp(self.nu*fbm_paths - 0.5*self.nu**2*t_grid**(2*self.H)))

        log_prices = np.log(self.S0) * np.ones((M, 1))

        for j in range(1, self.N + 1):

            log_prices += -0.5 * vol_paths[:, [j-1]]**2 * dt +  \
                 vol_paths[:, [j-1]] * (bm_paths[:, [j]] - bm_paths[:, [j-1]])

        return log_prices
    
    
    def get_option_price(self, strike, mc_runs, flag = 'c'):
        '''
        Returns European call option price based on Monte Carlo Simulations'''
        
        K = strike
        M = mc_runs
        stock_prices = np.exp(self.get_log_prices(M))
        
        if type(K) == int:
            if flag == 'c':
                payoffs = np.maximum(stock_prices - K, 0)
                
            elif flag == 'p':
                payoffs = np.maximum(K - stock_prices, 0)
                
            price = np.average(payoffs)
            
            std_price = np.std(payoffs)/np.sqrt(M)
            
        else:
            price = []
            std_price = []
            for k in K:
                if flag == 'c':
                    payoffs = np.maximum(stock_prices - k, 0)
                
                elif flag == 'p':
                    payoffs = np.maximum(k - stock_prices, 0)
                
                price.append(np.average(payoffs))
            
                std_price.append(np.std(payoffs)/np.sqrt(M))
                
        return price, std_price
    