import networkx as nx
import numpy as np
import pandas as pd
import powerlaw
from scipy.stats import bernoulli


def monte_carlo_resampling(min_length,
                           G,
                           n_lengths=5,
                           n_repetitions=10,
                           verbose=0):

    min_length = int(min_length)
    if isinstance(G, nx.classes.graph.Graph):
        max_length = int(len(G.nodes))
        degrees = pd.DataFrame(G.degree(), columns=['Nodes',
                                                    'Degree'])['Degree'].values
    else:
        max_length = len(G)
        degrees = G

    results = []

    for k in np.linspace(min_length, max_length, n_lengths):
        k = int(k)
        if verbose:
            print('length {}'.format(k))
        for r in range(n_repetitions):
            results.append(np.random.choice(degrees, size=k, replace=False))

    return results


def p_value(degrees, n_bootstrap_resampling=25, verbose=0):
    data = np.array(degrees)
    fit_AD = powerlaw.Fit(data, discrete=True, xmin_distance='Asquare')
    observed_AD = fit_AD.power_law.Asquare
    n = float(len(data))
    n_tail_AD_sample = float(len(data[data >= fit_AD.power_law.xmin]))
    non_pl_AD_sample = data[data < fit_AD.power_law.xmin]
    B_AD = bernoulli(n_tail_AD_sample / n)
    AD_distances = []

    m = 0
    if verbose:
        print('Starting bootstrap')
    while m < 50:
        bern_AD = B_AD.rvs(size=len(data))
        AD_distances.append(
            powerlaw.Fit(np.hstack(
                (fit_AD.power_law.generate_random(n=len(bern_AD[bern_AD == 1]),
                                                  estimate_discrete=True),
                 np.random.choice(non_pl_AD_sample,
                                  len(bern_AD[bern_AD == 0]),
                                  replace=True))),
                         discrete=True,
                         xmin_distance='Asquare',
                         verbose=False).power_law.Asquare)

        m = m + 1

    AD_distances = np.array(AD_distances)
    p_value = float(len(AD_distances[AD_distances > observed_AD])) / float(
        len(AD_distances))
    return p_value
