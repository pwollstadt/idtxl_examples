"""Estimation from discrete data using IDTxl."""
import numpy as np
from idtxl.estimators_jidt import JidtDiscreteCMI, JidtDiscreteTE
from idtxl.idtxl_utils import calculate_mi
from idtxl.data import Data
from idtxl.multivariate_te import MultivariateTE

# 1 Use core esimtators with data discretisation. Generate Gaussian test data
# and call JIDT Discrete estimators using the build-in discretization.
n = 1000
covariance = 0.4
corr_expected = covariance / (1 * np.sqrt(covariance**2 + (1-covariance)**2))
expected_mi = calculate_mi(corr_expected)
source_cor = np.random.normal(0, 1, size=n)  # correlated src
source_uncor = np.random.normal(0, 1, size=n)  # uncorrelated src
target = (covariance * source_cor +
          (1 - covariance) * np.random.normal(0, 1, size=n))

settings = {'discretise_method': 'equal', 'n_discrete_bins': 5}
est = JidtDiscreteCMI(settings)
cmi = est.estimate(source_cor, target, source_uncor)
print('Estimated CMI: {0:.5f}, expected CMI: {1:.5f}'.format(cmi, expected_mi))
settings['history_target'] = 1
est = JidtDiscreteTE(settings)
te = est.estimate(source_cor[1:n], target[0:n - 1])
print('Estimated TE: {0:.5f}, expected TE: {1:.5f}'.format(te, expected_mi))

# 2 Use network inference algorithms on discrete data.
n_procs = 5
alphabet_size = 5
data = Data(np.random.randint(0, alphabet_size, size=(n, n_procs)),
            dim_order='sp',
            normalise=False)  # don't normalize discrete data

# Initialise analysis object and define settings
network_analysis = MultivariateTE()
settings = {'cmi_estimator': 'JidtDiscreteCMI',
            'alph1': alphabet_size,  # provide initial alphabet size for
            'alph2': alphabet_size,  # discrete CMI estimator
            'alphc': alphabet_size,
            'max_lag_sources': 5,
            'min_lag_sources': 1}

# Run analysis and display results.
results = network_analysis.analyse_network(settings=settings, data=data)
results.print_edge_list(weights='max_te_lag', fdr=False)
