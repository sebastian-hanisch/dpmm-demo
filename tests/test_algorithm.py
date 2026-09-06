import math
from collections import Counter

import numpy as np
import pytest
from scipy import integrate, stats

from dp_algorithm import (
    _kl_beta,
    _kl_normal_isotropic,
    compute_elbo,
    run,
    update_components,
    update_responsibilities,
    update_stick_breaking,
)
from dp_scenario import generate_instance


def test_kl_beta_is_zero_for_identical_distributions():
    assert _kl_beta(2.0, 3.0, 2.0, 3.0) == pytest.approx(0.0, abs=1e-10)


def test_kl_beta_matches_numeric_integration():
    """KL(Beta(2,3) || Beta(1,1)) numerisch gegen scipy.integrate.quad geprüft."""
    a1, b1, a2, b2 = 2.0, 3.0, 1.0, 1.0

    def integrand(x):
        p = stats.beta.pdf(x, a1, b1)
        q = stats.beta.pdf(x, a2, b2)
        return p * np.log(p / q) if p > 0 else 0.0

    numeric, _ = integrate.quad(integrand, 1e-6, 1 - 1e-6)
    assert _kl_beta(a1, b1, a2, b2) == pytest.approx(numeric, rel=1e-4)


def test_kl_normal_is_zero_for_identical_distributions():
    m = np.array([1.0, -2.0])
    assert _kl_normal_isotropic(m, 0.5, m, 0.5, d=2) == pytest.approx(0.0, abs=1e-10)


def test_kl_normal_matches_numeric_integration():
    """KL(N((1,0.5), 0.7*I) || N((0,0), 1.3*I)) via Monte-Carlo-Schätzung geprüft
    (2D-Integration numerisch teuer, Monte Carlo mit vielen Stichproben ist hier die
    praktikablere unabhängige Kreuzprüfung)."""
    m1, s1_sq = np.array([1.0, 0.5]), 0.7
    m0, tau2 = np.array([0.0, 0.0]), 1.3
    analytic = _kl_normal_isotropic(m1, s1_sq, m0, tau2, d=2)

    rng = np.random.default_rng(0)
    samples = rng.normal(loc=m1, scale=np.sqrt(s1_sq), size=(500_000, 2))
    log_p = stats.multivariate_normal.logpdf(samples, mean=m1, cov=s1_sq * np.eye(2))
    log_q = stats.multivariate_normal.logpdf(samples, mean=m0, cov=tau2 * np.eye(2))
    monte_carlo = float(np.mean(log_p - log_q))
    assert analytic == pytest.approx(monte_carlo, rel=0.02)


def test_responsibilities_sum_to_one():
    instance = generate_instance(30, 3, 0.15, seed=1)
    result = run(instance.as_array(), alpha=1.0, sigma2=0.09, truncation=6, seed=1)
    resp = np.array(result.final_step.responsibilities)
    assert np.allclose(resp.sum(axis=1), 1.0)


def test_elbo_never_decreases_across_iterations():
    """Struktur-Invariante: Koordinatenaufstieg garantiert, dass die ELBO bei jedem
    Schritt nicht sinkt - analog zu gmm-demos Log-Likelihood-Monotonie."""
    for seed in range(10):
        instance = generate_instance(60, 3, 0.2, seed=seed)
        result = run(instance.as_array(), alpha=0.5, sigma2=0.09, truncation=8, seed=seed)
        elbos = [s.elbo for s in result.steps]
        for i in range(len(elbos) - 1):
            assert elbos[i + 1] >= elbos[i] - 1e-6, f"seed {seed}: ELBO decreased at step {i}"


def test_run_produces_valid_labels_at_every_step():
    instance = generate_instance(30, 3, 0.15, seed=1)
    result = run(instance.as_array(), alpha=1.0, sigma2=0.09, truncation=6, seed=1)
    for step in result.steps:
        assert len(step.hard_labels) == 30
        assert step.n_active_components == len(set(step.hard_labels))


def test_truncation_caps_the_active_component_count():
    """Struktur-Invariante: die Trunkierungsgrenze T ist eine harte Obergrenze fuer die
    Anzahl aktiver Komponenten, unabhaengig von alpha."""
    instance = generate_instance(90, 6, 0.15, seed=1)
    result = run(instance.as_array(), alpha=5.0, sigma2=0.09, truncation=4, seed=1)
    assert result.final_step.n_active_components <= 4


def test_converged_flag_reflects_elbo_tolerance():
    instance = generate_instance(60, 3, 0.15, seed=1)
    result = run(instance.as_array(), alpha=0.3, sigma2=0.09, truncation=8, seed=1, max_iter=100)
    assert result.converged
    assert not result.truncated


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_matches_sklearn_bayesian_gaussian_mixture_qualitatively(seed):
    """Grober, QUALITATIVER Kreuzvergleich gegen sklearn.mixture.BayesianGaussianMixture
    (volle statt sphaerischer/bekannter Kovarianz - kein exaktes Zahlen-Match zu
    erwarten, aber beide sind jetzt tatsaechlich Variational Inference fuer eine
    DP-Mixture): bei klar getrennten Gruppen sollten beide ungefaehr dieselbe
    Groessenordnung an Komponenten finden."""
    mixture = pytest.importorskip("sklearn.mixture")

    instance = generate_instance(90, 3, 0.15, seed=seed)
    data = instance.as_array()
    ours = run(data, alpha=0.3, sigma2=0.09, truncation=10, seed=seed)

    theirs = mixture.BayesianGaussianMixture(
        n_components=10, weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=0.3, random_state=seed, max_iter=200,
    ).fit(data)
    theirs_labels = theirs.predict(data)
    theirs_k = len(set(theirs_labels.tolist()))

    assert abs(ours.final_step.n_active_components - theirs_k) <= 2


# --- Exakte Verifikation auf einer Spielzeug-Instanz -----------------------------
#
# Kern-Korrektheitsnachweis, UNABHAENGIG von der Inferenzmethode: auf einer winzigen
# Instanz (n=5) wird die exakte MAP-Partition durch Enumeration ALLER moeglichen
# Partitionen berechnet (Ewens-Sampling-Formel fuer den CRP-Prior, sequenzielle
# Praediktiv-Produkte fuer die marginale Likelihood je Cluster). Frueher (Collapsed-
# Gibbs-Sampler) wurde das gegen die EMPIRISCHE VERTEILUNG vieler MCMC-Laeufe
# verglichen; Variational Inference liefert stattdessen einen deterministischen
# Punktschaetzer - hier wird direkt geprueft, ob dieser Punktschaetzer die exakte
# MAP-Partition trifft, ein Vergleich, den das samplingbasierte Verfahren so nicht
# bieten konnte.


def _predictive_params(member_sum, n_members, sigma2, mu0, tau2):
    """Normal-Normal-Posterior-Praediktivverteilung - eine Eigenschaft des MODELLS
    (Cluster-Mittel analytisch herausintegriert), unabhaengig von der Inferenzmethode.
    Lebt bewusst NUR hier (nicht mehr in dp_algorithm.py, das jetzt Variational
    Inference statt eines Collapsed Samplers implementiert), da sie ausschliesslich
    fuer die exakte Ground-Truth-Berechnung gebraucht wird."""
    mu0 = np.asarray(mu0, dtype=float)
    post_var = 1.0 / (1.0 / tau2 + n_members / sigma2)
    post_mean = post_var * (mu0 / tau2 + np.asarray(member_sum, dtype=float) / sigma2)
    predictive_var = post_var + sigma2
    return post_mean, predictive_var


def _log_gaussian_isotropic(x, mean, var):
    diff = np.asarray(x, dtype=float) - mean
    d = len(mean)
    return -0.5 * (d * np.log(2 * np.pi * var) + float(diff @ diff) / var)


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
    log_p = len(block_sizes) * math.log(alpha)
    for m in block_sizes:
        log_p += math.lgamma(m)
    log_p -= math.lgamma(alpha + n) - math.lgamma(alpha)
    return log_p


def _log_cluster_marginal(points, sigma2, mu0, tau2):
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


def test_variational_inference_converges_to_exact_map_partition_on_toy_instance():
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
    exact_map_partition = partitions[int(np.argmax(log_posteriors))]
    exact_map_key = _canonical(exact_map_partition)

    agreements = 0
    n_restarts = 20
    for seed in range(n_restarts):
        result = run(np.array(data), alpha, sigma2, truncation=5, seed=seed, mu0=mu0, tau2=tau2)
        labels = result.final_labels
        blocks = {}
        for idx, lab in enumerate(labels):
            blocks.setdefault(lab, []).append(idx)
        if _canonical(list(blocks.values())) == exact_map_key:
            agreements += 1

    # Variational Inference ist ein LOKALER Optimierer (abhaengig von der Initialisierung) -
    # anders als beim MCMC-Sampler wird hier keine Uebereinstimmung mit der stationaeren
    # Verteilung erwartet, sondern dass die MEISTEN Neustarts die exakte MAP-Partition
    # finden (ein direkter, aussagekraeftiger "wie gut ist die Approximation"-Nachweis).
    assert agreements / n_restarts > 0.6, f"nur {agreements}/{n_restarts} Neustarts trafen die exakte MAP-Partition"
