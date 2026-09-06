import math
from collections import Counter

import numpy as np
import pytest

from dp_algorithm import (
    _log_gaussian_isotropic,
    _predictive_params,
    conditional_distribution,
    crp_log_prior_existing,
    crp_log_prior_new,
    run,
)
from dp_scenario import generate_instance


def test_predictive_params_hand_computed():
    """mu0=(0,0), tau2=4, sigma2=1, zwei Punkte mit Summe (2,0): von Hand:
    post_var = 1/(1/4 + 2/1) = 4/9, post_mean = post_var*(0 + (2,0)/1) = (8/9, 0),
    predictive_var = post_var + sigma2 = 4/9 + 1."""
    mu0 = np.array([0.0, 0.0])
    tau2 = 4.0
    sigma2 = 1.0
    member_sum = np.array([2.0, 0.0])

    mean, var = _predictive_params(member_sum, 2, sigma2, mu0, tau2)

    assert var == pytest.approx(4.0 / 9.0 + 1.0)
    assert mean[0] == pytest.approx(8.0 / 9.0)
    assert mean[1] == pytest.approx(0.0)


def test_predictive_params_with_zero_members_is_prior_predictive():
    mu0 = np.array([1.0, -2.0])
    tau2, sigma2 = 3.0, 0.5
    mean, var = _predictive_params(np.zeros(2), 0, sigma2, mu0, tau2)
    assert mean[0] == pytest.approx(1.0)
    assert mean[1] == pytest.approx(-2.0)
    assert var == pytest.approx(tau2 + sigma2)


def test_crp_log_priors_hand_computed():
    """alpha=2, n-1=5: bestehendes Cluster mit 3 ANDEREN Mitgliedern -> ln(3/7); neues
    Cluster -> ln(2/7)."""
    alpha, n_minus_1 = 2.0, 5
    existing = crp_log_prior_existing(3, n_minus_1, alpha)
    new = crp_log_prior_new(n_minus_1, alpha)
    assert existing == pytest.approx(math.log(3.0 / 7.0))
    assert new == pytest.approx(math.log(2.0 / 7.0))


def test_conditional_distribution_probabilities_sum_to_one():
    counts = {0: 3, 1: 2}
    sums = {0: np.array([3.0, 0.0]), 1: np.array([-4.0, 0.0])}
    x_i = np.array([0.1, 0.1])
    candidates, probs = conditional_distribution(
        x_i, counts, sums, alpha=1.0, sigma2=0.25, mu0=np.zeros(2), tau2=4.0, n_minus_1=5
    )
    assert len(candidates) == 3  # zwei bestehende Cluster + "neues Cluster"
    assert probs.sum() == pytest.approx(1.0)
    assert (probs >= 0).all()


def test_run_produces_valid_labels_at_every_sweep():
    instance = generate_instance(30, 3, 0.15, seed=1)
    result = run(instance.as_array(), alpha=1.0, sigma2=0.09, seed=1, n_sweeps=10)
    for sweep in result.sweeps:
        assert len(sweep.labels) == 30
        assert sweep.n_clusters == len(set(sweep.labels))
        assert set(sweep.labels) == set(range(sweep.n_clusters))


def test_first_sweep_starts_from_all_singletons():
    instance = generate_instance(12, 3, 0.15, seed=2)
    result = run(instance.as_array(), alpha=1.0, sigma2=0.09, seed=2, n_sweeps=1)
    assert result.sweeps[0].n_clusters == 12


# --- Exakte Verifikation auf einer Spielzeug-Instanz -----------------------------
#
# Kern-Korrektheitsnachweis: auf einer winzigen Instanz (n=5) werden ALLE möglichen
# Partitionen enumeriert und ihre exakte Posterior-Wahrscheinlichkeit berechnet (Ewens-
# Sampling-Formel für den CRP-Prior, sequenzielle Prädiktiv-Produkte für die marginale
# Likelihood je Cluster - beide UNABHÄNGIG von der internen Sampler-Logik implementiert,
# auch wenn dieselbe `_predictive_params`-Formel wiederverwendet wird). Dagegen wird die
# empirische Verteilung vieler unabhängiger Sampler-Läufe verglichen.


def _all_partitions(elements):
    """Alle Mengenpartitionen einer Liste - klassischer rekursiver Algorithmus, nur für
    sehr kleine n verwendet (Bell-Zahl viele: B_5 = 52)."""
    if len(elements) == 1:
        yield [[elements[0]]]
        return
    first, rest = elements[0], elements[1:]
    for smaller in _all_partitions(rest):
        for i in range(len(smaller)):
            yield smaller[:i] + [[first] + smaller[i]] + smaller[i + 1 :]
        yield [[first]] + smaller


def _log_ewens_prior(block_sizes, alpha, n):
    """Exakte CRP-Prior-Wahrscheinlichkeit (log) einer Partition mit den gegebenen
    Blockgrößen - die Ewens-Sampling-Formel, unabhängig von den sequenziellen
    Ein-Punkt-CRP-Updates des Samplers."""
    log_p = len(block_sizes) * math.log(alpha)
    for m in block_sizes:
        log_p += math.lgamma(m)  # (m-1)! = Gamma(m)
    log_p -= math.lgamma(alpha + n) - math.lgamma(alpha)
    return log_p


def _log_cluster_marginal(points, sigma2, mu0, tau2):
    """Marginale Likelihood eines Clusters (Cluster-Mittel herausintegriert) über die
    SEQUENZIELLE Anwendung derselben Posterior-Prädiktivformel wie im Sampler - die
    Exchangeability der CRP garantiert, dass die gemeinsame Dichte unabhängig von der
    gewählten Reihenfolge ist."""
    log_p = 0.0
    running_sum = np.zeros(2)
    for j, x in enumerate(points):
        mean, var = _predictive_params(running_sum, j, sigma2, mu0, tau2)
        log_p += _log_gaussian_isotropic(np.asarray(x), mean, var)
        running_sum = running_sum + np.asarray(x)
    return log_p


def _exact_log_posterior(partition, data, alpha, sigma2, mu0, tau2):
    block_sizes = [len(b) for b in partition]
    log_p = _log_ewens_prior(block_sizes, alpha, len(data))
    for block in partition:
        pts = [data[i] for i in block]
        log_p += _log_cluster_marginal(pts, sigma2, mu0, tau2)
    return log_p


def _canonical(partition):
    return frozenset(frozenset(b) for b in partition)


def test_sampler_stationary_distribution_matches_exact_posterior_on_toy_instance():
    alpha, sigma2 = 1.0, 0.3
    mu0, tau2 = (0.0, 0.0), 4.0
    data = [
        (0.0, 0.0), (0.15, -0.1),
        (3.0, 0.0), (3.1, 0.1),
        (1.5, 2.5),
    ]
    n = len(data)

    partitions = list(_all_partitions(list(range(n))))
    log_posteriors = np.array(
        [_exact_log_posterior(p, data, alpha, sigma2, mu0, tau2) for p in partitions]
    )
    log_posteriors -= log_posteriors.max()
    exact_probs = np.exp(log_posteriors)
    exact_probs /= exact_probs.sum()
    exact_by_key = {_canonical(p): prob for p, prob in zip(partitions, exact_probs)}

    n_chains = 300
    counts = Counter()
    for chain in range(n_chains):
        result = run(np.array(data), alpha, sigma2, seed=1000 + chain, n_sweeps=25, mu0=mu0, tau2=tau2)
        labels = result.final_labels
        blocks = {}
        for idx, lab in enumerate(labels):
            blocks.setdefault(lab, []).append(idx)
        counts[_canonical(list(blocks.values()))] += 1

    empirical_probs = {k: v / n_chains for k, v in counts.items()}

    exact_map = max(exact_by_key, key=exact_by_key.get)
    empirical_map = max(empirical_probs, key=empirical_probs.get)
    assert exact_map == empirical_map, (
        f"exact MAP {exact_map} (p={exact_by_key[exact_map]:.3f}) != "
        f"empirical MAP {empirical_map} (p={empirical_probs[empirical_map]:.3f})"
    )
    assert abs(exact_by_key[exact_map] - empirical_probs[exact_map]) < 0.2
