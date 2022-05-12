# Compressed Time Series Regression (v 0.1)

CompressedTSRegression estimates recursively a compressed linear 
switching state space model for a univariate target time series `y` given 
high-dimensional `X` (to be compressed) and non-compressible predictors `exog`. 
This is achieved by first sampling compression regimes from a proposal distribution, 
followed by recursive filtering conditional on each compression using forgetting factors. 
Finally, unconditional predictions are obtained 
using dynamic Bayesian model averaging over regime-conditional point predictions.

## Usage
```python
import numpy as np
from scipy.sparse import csr_matrix
from ctsreg import CompressedTSRegression

y = np.random.standard_normal((200,1))
X = csr_matrix(np.random.standard_normal((200,1000)))
exog = np.ones((200,1))

reg = CompressedTSRegression()
fitted = reg.fit_eval(X, exog, y)

fitted.simple_preds
fitted.weighted_preds
```

## Parameters

### periods_ahead : int, default=1,
Number of periods y is ahead of X (i.e. the forecast horizon).

### eval_periods : int, default=100
Number of periods to use for evaluation. This determines
    how many predicted values are returned. Needs to exceed the sample size.
### subspace_dim_limit : int, default=100
    The maximum number of dimensions of the compression subspace. Larger values
    tend to increase accuracy but increase RAM and CPU intensity.
### s_draw_lb : float, default=0.1
    Lower bound for uniform distribution generating compression proposal parameter s. 
### s_draw_ub : float, default=0.9
    Upper bound for uniform distribution generating compression proposal parameter s.
### lambda_value : float, default=0.99
    Forgetting factor for the state covariance when Kalman filtering
### kappa_value : float, default=0.99
    Forgetting/decay factor for the plug-in exponentially-weighted moving average estimation
    of the observation variance when Kalman filtering.
### alpha_grid_size : int, default=10
    Number of grid elements when tuning the forgetting factor for dynamic Bayesian model 
    averaging.
### model_posterior_eps : float, default=0.001
    Small bias added to posterior model probabilities to prevent numerical underflow.
### n_jobs : int, default=1
    Number of parallel workers to use. If RAM permits, set to number of logical processors
    to maximise estimation speed.
