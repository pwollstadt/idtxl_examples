import numpy as np

from idtxl.estimators_jidt import JidtKraskovCMI, JidtKraskovTE
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data

## Use IDTxl's core esitmator if lags/embeddings are known

# Generate high-dimensional example processes
n = 1000
source_dim = 3
cond_dim = 2
target = np.random.randn(n)
source = np.random.randn(n, source_dim)
conditional = np.random.randn(n, cond_dim)

settings = {}
est = JidtKraskovCMI(settings)
cmi = est.estimate(source, target, conditional)
print(f'CMI estimate: {cmi:.4f}')

## Use IDTxl's network inference algorithm to optimize lags/embeddings

# Generate test data, we assume that the processes represent sources, a target,
# and additional processes, we want to condition on. For the conditioning, we
# have to provide tuples of past variables plus a lag.
data = Data(np.random.randn(5, n), dim_order='ps')
target = 0
sources = [1, 2]
cond_1_ind = 3
cond_2_ind = 4
cond_1_lag = 1
cond_2_lag = 1

# Initialise analysis object and define settings
network_analysis = MultivariateTE()
settings = {
    'cmi_estimator': 'JidtKraskovCMI',
    'add_conditionals': [(cond_1_ind, cond_1_lag), (cond_2_ind, cond_2_lag)],
    'max_lag_sources': 3,
    'min_lag_sources': 1}

# c) Run analysis
results = network_analysis.analyse_single_target(
    settings=settings, data=data, target=target, sources=sources)
results.print_edge_list(weights='max_te_lag', fdr=False)
