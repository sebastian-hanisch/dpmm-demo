"""Rand-Index (from scratch, wiederverwendetes Muster) sowie der Alpha-Sensitivitäts-
Vergleich - das DPMM-Äquivalent zu dbscan-demos eps-Sweep: zeigt, wie stark die gefundene
Clusteranzahl vom Konzentrationsparameter alpha abhängt."""

import numpy as np

from dp_algorithm import run


def rand_index(true_labels, pred_labels):
    """Anteil der Punktpaare, bei denen beide Partitionen übereinstimmen (beide im
    selben Cluster oder beide in unterschiedlichen)."""
    true_arr = np.asarray(true_labels)
    pred_arr = np.asarray(pred_labels)
    n = len(true_arr)
    if n < 2:
        return 1.0

    iu = np.triu_indices(n, k=1)
    same_true = (true_arr[:, None] == true_arr[None, :])[iu]
    same_pred = (pred_arr[:, None] == pred_arr[None, :])[iu]
    return float(np.mean(same_true == same_pred))


def alpha_comparison(data, true_k, alpha_values, sigma2, seed, n_sweeps):
    """Gefundene vs. wahre Clusteranzahl über mehrere alpha-Werte, jeweils bei
    Sampler-Endstand - macht direkt sichtbar, wie stark alpha die entdeckte
    Clusterzahl beeinflusst, ohne dass irgendwo k selbst vorgegeben wurde."""
    results = {}
    for alpha in alpha_values:
        result = run(data, alpha, sigma2, seed, n_sweeps=n_sweeps)
        results[alpha] = {
            "found_k": result.final_sweep.n_clusters,
            "true_k": true_k,
        }
    return results
