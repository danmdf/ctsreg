# ctsreg (version 0.1, 12/05/2022)

import numpy as np
from scipy.stats import norm
from scipy.sparse import csr_matrix
from decimal import *
import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import diags

class CompressedTSRegression:
    """
    Compressed Time Series Regression.
    CompressedTSRegression estimates recursively a compressed linear 
    switching state space model for a univariate target time series y given 
    high-dimensional X (to be compressed) and non-compressible predictors exog. 
    This is achieved by first sampling compression regimes from a proposal distribution, 
    followed by recursive filtering conditional on each compression using forgetting factors. 
    Finally, unconditional predictions are obtained 
    using dynamic Bayesian model averaging over regime-conditional point predictions.

    Parameters
    ----------
    periods_ahead : int, default=1,
        Number of periods y is ahead of X (i.e. the forecast horizon).
    eval_periods : int, default=100
        Number of periods to use for evaluation. This determines
        how many predicted values are returned. Needs to exceed the sample size.
    subspace_dim_limit : int, default=100
        The maximum number of dimensions of the compression subspace. Larger values
        tend to increase accuracy but increase RAM and CPU intensity.
    s_draw_lb : float, default=0.1
        Lower bound for uniform distribution generating compression proposal parameter s. 
    s_draw_ub : float, default=0.9
        Upper bound for uniform distribution generating compression proposal parameter s.
    lambda_value : float, default=0.99
        Forgetting factor for the state covariance when Kalman filtering
    kappa_value : float, default=0.99
        Forgetting/decay factor for the plug-in exponentially-weighted moving average estimation
        of the observation variance when Kalman filtering.
    alpha_grid_size : int, default=10
        Number of grid elements when tuning the forgetting factor for dynamic Bayesian model 
        averaging.
    model_posterior_eps : float, default=0.001
        Small bias added to posterior model probabilities to prevent numerical underflow.
    n_jobs : int, default=1
        Number of parallel workers to use. If RAM permits, set to number of logical processors
        to maximise estimation speed.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>> from ctsreg import CompressedTSRegression

    >>> y = np.random.standard_normal((200,1))
    >>> X = csr_matrix(np.random.standard_normal((200,1000)))
    >>> exog = np.ones((200,1))

    >>> reg = CompressedTSRegression()
    >>> fitted = reg.fit_eval(X, exog, y)
    >>> fitted.simple_preds
    >>> fitted.weighted_preds
    """

    def __init__(
        self,
        *,
        periods_ahead=1,
        eval_periods=100,
        subspace_dim_limit=100,
        s_draw_lb=0.1,
        s_draw_ub=0.9,
        lambda_value=0.99,
        kappa_value=0.99,
        alpha_grid_size=10,
        model_posterior_eps=0.001,
        n_jobs=1

    ):
        self.periods_ahead = periods_ahead
        self.eval_periods=eval_periods
        self.subspace_dim_limit = subspace_dim_limit
        self.s_draw_lb = s_draw_lb
        self.s_draw_ub = s_draw_ub
        self.lambda_value = lambda_value
        self.kappa_value = kappa_value
        self.alpha_grid_size = alpha_grid_size
        self.model_posterior_eps = model_posterior_eps  
        self.n_jobs = n_jobs
  

    def fit_eval(self, X, exog, y):
        """
        Fit Compressed Time Series Regression for evaluation.
        Parameters
        ----------
        X : scipy.sparse matrix of shape (T, D)
        exog : numpy array of shape (T, D')
        y : numpy array of shape (T, 1)

        Returns
        -------
        self : object
            Fitted Estimator
        """

        # check that no NA values
        assert (~np.isnan(X.toarray())).all()

        self.T = X.shape[0]
        self.D = X.shape[1]

        # link number of hypothesised compression regimes to D
        self.n_models = min(self.T, self.D) - int(2*round(np.log(self.D), 0)) 

        # run recursive filtering of compression regime models in parallel
        preds_list, raw_weights_list = zip(*Parallel(n_jobs=self.n_jobs)(delayed(self._compressed_filter_worker)(X, exog, y) for i in range(self.n_models)))
        
        self.eval_preds = np.concatenate(preds_list, axis=1)
        self.y_density_evals = np.concatenate(raw_weights_list, axis=1)

        # tune "a" values (regime averaging forgetting factor) to get best in-sample loss
        a_vals, losses = zip(*Parallel(n_jobs=self.n_jobs)(delayed(self._tune_a)(self.y_density_evals, self.eval_preds, y, a) for a in np.linspace(0, 1, self.alpha_grid_size)))
        self.best_alpha = a_vals[np.array(losses).argmin()]

        # get simple average evaluation predictions
        self.simple_preds = self.eval_preds[-self.eval_periods:].mean(axis=1)

        # get weighted average evaluation predictions (using periods_ahead-lag weights and tuned "a" value)
        self.weighted_preds = np.multiply(self._get_model_weights(self.y_density_evals, self.best_alpha)[:- self.periods_ahead][-self.eval_periods:], self.eval_preds[-self.eval_periods:]).sum(axis=1)

        return self

    # TODO add fit_predict(self, X_train, exog_train, y_train, X_test, exog_test)

    def _compressed_filter_worker(self, X, exog, y):
        """Worker function to be parallelised for estimating a single compression regime model."""

        # STEP 1: DRAW COMPRESSION MATRIX PHI
        # DRAW PHI (no real look ahead bias, because we only need to know rough order of magnitude of D and N in advance)
        k = np.random.randint(low=int(2*round(np.log(self.D), 0)), high=self.subspace_dim_limit)
        s = np.random.uniform(self.s_draw_lb, self.s_draw_ub)

        raw_phi = np.random.choice([-(1/s)**(1/2), 0, (1/s)**(1/2)],
                                   size=self.D * k,
                                   p=[s**2, 2*(1 - s)*s, (1 - s)**2]).reshape(self.D, k)

        # Gram-Schmidt orthonormalisation
        phi = csr_matrix(np.linalg.qr(raw_phi)[0])

        # STEP 2: COMPRESS
        X_phi = X @ phi

        # NORMALISE
        X_phi = (X_phi.T @ diags(1/X.sum(axis=1).A.ravel())).T

        # STEP 3: RUN FILTER
        eval_preds, raw_weights = self._kalman_filter(X_phi, exog, y)

        return eval_preds, raw_weights


    def _kalman_filter(self, X_phi, exog, y):
        """Recursive estimation of a single compression regime model given compression X_phi."""

        X = np.concatenate((exog, X_phi.toarray()), axis=1)

        # get dimension
        KF_D = X.shape[1]

        # set priors (avoiding look-ahead bias)
        theta_mean_prior = np.zeros((1, KF_D))
        theta_var_prior_const = y[:-self.eval_periods - self.periods_ahead].var().reshape(-1)
        theta_var_prior_x = y[:-self.eval_periods - self.periods_ahead].var()/(X[:-self.eval_periods - self.periods_ahead].var(axis=0)[1:])
        theta_var_prior = np.diag(np.concatenate([theta_var_prior_const, theta_var_prior_x]))
        sigma_prior = y[:-self.eval_periods - self.periods_ahead].var()

        # storage
        theta_mean_pred = np.zeros((self.T, KF_D))  # predicted state mean
        theta_mean_update = np.zeros((self.T, KF_D))  # updated state mean
        theta_var_pred = np.zeros((self.T, KF_D, KF_D))  # predicted state var
        theta_var_update = np.zeros((self.T, KF_D, KF_D))  # updated state var
        sigma_pred = np.zeros((self.T, 1))  # predicted observation var
        e = np.zeros((self.T, 1))  # residual
        y_dens = np.zeros((self.T, 1))  # predictive density evaluated at actual y

        # Kalman filter: t == 0
        # predict step
        theta_mean_pred[0, :] = theta_mean_prior  # predict state mean
        theta_var_pred[0, :, :] = (1/self.lambda_value)*theta_var_prior  # predict state var
        sigma_pred[0] = sigma_prior  # predict observation var

        # update step
        e[0] = y[0] - X[0, :] @ theta_mean_pred[0, :].T  # obtain residual
        KG = (theta_var_pred[0, :, :] @ X[0, :].T) / (X[0, :] @ theta_var_pred[0, :, :] @ X[0, :].T + sigma_pred[0])  # Kalman gain
        theta_mean_update[0, :] = theta_mean_pred[0, :] + KG*e[0]  # update state mean
        theta_var_update[0, :, :] = (np.identity(KF_D) - KG.reshape(-1, 1) @ X[0, :].reshape(1, -1)) @ theta_var_pred[0, :, :]  # update state var

        # evaluate predictive density at actual y for model averaging (NOTE that argument of norm is SD, not VAR)
        y_dens[0] = norm(X[0, :] @ theta_mean_pred[0, :].T, (sigma_pred[0] + X[0, :] @ theta_var_pred[0, :, :] @ X[0, :].T)**(1/2)).pdf(y[0])

        # Kalman filter: all other periods
        for t in range(1, self.T):
            # predict step
            # predict state mean
            theta_mean_pred[t, :] = theta_mean_update[t-1, :]
            theta_var_pred[t, :, :] = (1/self.lambda_value)*theta_var_update[t-1, :, :]  # predict state var
            sigma_pred[t] = self.kappa_value*sigma_pred[t-1] + (1-self.kappa_value)*e[t-1]**2  # predict observation var

            # update step
            e[t] = y[t] - X[t, :] @ theta_mean_pred[t, :].T  # obtain residual
            KG = (theta_var_pred[t, :, :] @ X[t, :].T) / (X[t, :] @ theta_var_pred[t, :, :] @ X[t, :].T + sigma_pred[t])  # Kalman gain
            theta_mean_update[t, :] = theta_mean_pred[t, :] + KG*e[t]  # update state mean
            theta_var_update[t, :, :] = (np.identity(KF_D) - KG.reshape(-1, 1) @ X[t, :].reshape(1, -1)) @ theta_var_pred[t, :, :]  # update state var

            # evaluate predictive density at actual y for model averaging (NOTE that input is SD, not VAR)
            y_dens[t] = norm(X[t, :] @ theta_mean_pred[t, :].T, (sigma_pred[t] + X[t, :] @ theta_var_pred[t, :, :] @ X[t, :].T)**(1/2)).pdf(y[t])
            
            if y_dens[t] == 0:  # diagnose print if numerical underflow
                print("DENSITY == 0 EVAL! Period {}, y_pred {}, sigma {}, xRx2 {}".format(t, sigma_pred[t], X[t, :] @ theta_var_pred[t, :, :] @ X[t, :].T))

        # now obtain eval predictions (bearing in mind the prediction horizon)
        eval_preds = []
        for t in range(self.T):
            if t < self.periods_ahead:
                # NOTE: use prior
                eval_preds.append(X[t, :] @ theta_mean_prior.flatten())

            else:
                # NOTE: use periods_ahead-period old estimate
                eval_preds.append(
                    X[t, :] @ theta_mean_update[t - self.periods_ahead, :])

        eval_preds = np.array(eval_preds).reshape(-1, 1)

        return eval_preds, y_dens.reshape(-1, 1)


    def _get_model_weights(self, y_density_evals, a_val):
        """Implements dynamic Bayesian averaging given a set of density evaluations
            and a forgetting factor (a_val)"""

        # init model weights
        pi_prior = np.ones(self.n_models)/self.n_models  # i.e. uniform prior over models
        pi_pred = np.zeros((self.T, self.n_models))
        pi_update = np.zeros((self.T, self.n_models))

        # now do the same loop with the best value
        for t in range(self.T):
            #print("begin filter: period {}".format(t))
            # loop over models
            for model in range(self.n_models):
                if t == 0:
                    # Last term for numerical stability, as in Raftery
                    pi_pred[t, model] = pi_prior[model]**a_val + \
                        self.model_posterior_eps/self.n_models
                else:
                    # Last term for numerical stability, as in Raftery
                    pi_pred[t, model] = pi_update[t-1, model]**a_val + \
                        self.model_posterior_eps/self.n_models

            # normalise to sum to 1
            pi_pred[t, :] = pi_pred[t, :]/pi_pred[t, :].sum()

            for model in range(self.n_models):
                # updated using predictive density evaluation (done in Kalman function)

                # catch numerical issues here!
                if y_density_evals[t, model] == 0:
                    print("ZERO DENSITY EVAL! PERIOD {}, proj {}".format(t, model))

                pi_update[t, model] = pi_pred[t, model] * \
                    y_density_evals[t, model]

            # normalise to sum to 1
            pi_update[t, :] = pi_update[t, :]/pi_update[t, :].sum()

        return pi_update


    def _tune_a(self, y_density_evals, eval_preds, y, a_val):
        """worker to parallelise the tuning step"""
        return a_val, ((np.multiply(self._get_model_weights(y_density_evals, a_val)[:-self.eval_periods- self.periods_ahead], eval_preds[self.periods_ahead:-self.eval_periods]).sum(axis=1) - y[self.periods_ahead:-self.eval_periods])**2).mean()


