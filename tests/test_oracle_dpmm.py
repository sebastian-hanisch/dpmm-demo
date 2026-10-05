"""Orakel-Tests für die Variational Inference (unabhängige Rechenwege, ohne den eigenen Update-Code):

1. ELBO per Monte-Carlo-Erwartung unter q (Beta- und Normalverteilungen gezogen, Dichten aus scipy)
   statt über die geschlossenen KL-Formeln des Moduls.
2. ELBO <= exakte Log-Evidenz des trunkierten Modells (Aufzählung aller T^n Zuordnungen,
   Beta-Momente für die Stick-Breaking-Gewichte, Cluster-Mittelwerte analytisch herausintegriert).
3. Lokale Optimalität der drei Block-Updates: bei festem Rest erhöht keine Störung der Parameter die ELBO.
4. Rand-Index gegen sklearn.metrics.rand_score."""

import itertools

import numpy as np
import pytest
from scipy import stats
from scipy.special import betaln, logsumexp

import dp_algorithm as A
from dp_evaluation import rand_index
from dp_scenario import generate_instance


def _mc_elbo(data, a, b, m, v, phi, alpha, s2, mu0, tau2, rng, n_samples=40000):
    n, d = data.shape
    T = len(m)
    beta = np.clip(stats.beta.rvs(a, b, size=(n_samples, T - 1), random_state=rng), 1e-300, 1 - 1e-15)
    log_pi = np.empty((n_samples, T))
    cum = np.zeros(n_samples)
    for t in range(T - 1):
        log_pi[:, t] = np.log(beta[:, t]) + cum
        cum = cum + np.log1p(-beta[:, t])
    log_pi[:, T - 1] = cum
    mu = m[None] + np.sqrt(v)[None, :, None] * rng.standard_normal((n_samples, T, d))
    log_pv = stats.beta.logpdf(beta, 1.0, alpha).sum(1)
    log_qv = stats.beta.logpdf(beta, a, b).sum(1)
    log_pm = (-0.5 * d * np.log(2 * np.pi * tau2) - 0.5 * ((mu - mu0) ** 2).sum(2) / tau2).sum(1)
    log_qm = (-0.5 * d * np.log(2 * np.pi * v)[None] - 0.5 * ((mu - m[None]) ** 2).sum(2) / v[None]).sum(1)
    ll = np.zeros(n_samples)
    for t in range(T):
        sq = ((data[None, :, :] - mu[:, t, None, :]) ** 2).sum(2)
        lg = -0.5 * d * np.log(2 * np.pi * s2) - 0.5 * sq / s2
        ll += (phi[:, t][None, :] * (log_pi[:, t, None] + lg)).sum(1)
    entropy = -(phi * np.log(np.clip(phi, 1e-300, 1))).sum()
    f = log_pv + log_pm + ll - log_qv - log_qm + entropy
    return f.mean(), f.std() / np.sqrt(n_samples)


def _exact_log_evidence(data, T, alpha, s2, mu0, tau2):
    n = len(data)
    logs = []
    for z in itertools.product(range(T), repeat=n):
        z = np.array(z)
        counts = np.bincount(z, minlength=T)
        lp = 0.0
        for t in range(T - 1):
            lp += betaln(1 + counts[t], alpha + counts[t + 1 :].sum()) - betaln(1, alpha)
        for t in range(T):
            running = np.zeros(2)
            for j, x in enumerate(data[z == t]):
                post_var = 1 / (1 / tau2 + j / s2)
                post_mean = post_var * (np.asarray(mu0) / tau2 + running / s2)
                lp += stats.multivariate_normal(post_mean, (post_var + s2) * np.eye(2)).logpdf(x)
                running = running + x
        logs.append(lp)
    return logsumexp(logs)


def test_elbo_matches_monte_carlo_and_is_below_exact_evidence():
    rng = np.random.default_rng(3)
    tau2, mu0 = 16.0, np.zeros(2)
    for trial in range(6):
        data = rng.normal(size=(5, 2)) * rng.uniform(0.5, 2.0)
        alpha, s2, T = float(rng.uniform(0.6, 3.0)), float(rng.uniform(0.1, 1.0)), 3
        result = A.run(data, alpha, s2, T, trial, max_iter=6, tol=0.0, mu0=tuple(mu0), tau2=tau2)

        step = result.steps[-1]
        counts = np.array(result.steps[-2].responsibilities).sum(0)
        comp_var = 1.0 / (1.0 / tau2 + counts / s2)
        mean, se = _mc_elbo(
            data, np.array(step.stick_a), np.array(step.stick_b), np.array(step.means), comp_var,
            np.array(step.responsibilities), alpha, s2, mu0, tau2, rng,
        )
        assert abs(mean - step.elbo) < 5 * se + 1e-3

        evidence = _exact_log_evidence(data, T, alpha, s2, mu0, tau2)
        assert max(s.elbo for s in result.steps) <= evidence + 1e-9


def test_block_updates_are_locally_optimal():
    rng = np.random.default_rng(7)
    tau2, mu0 = 16.0, np.zeros(2)
    for trial in range(8):
        data = generate_instance(40, int(rng.integers(2, 5)), 0.2, trial).as_array()
        T, alpha, s2 = int(rng.integers(2, 7)), float(rng.uniform(0.05, 5)), float(rng.uniform(0.05, 1.0))
        phi = rng.dirichlet(np.ones(T), size=len(data))
        a, b = A.update_stick_breaking(phi, alpha)
        m, v = A.update_components(data, phi, s2, tuple(mu0), tau2)
        base = A.compute_elbo(data, a, b, m, v, phi, alpha, s2, tuple(mu0), tau2)
        for _ in range(15):
            a2 = np.maximum(a + rng.normal(scale=0.3, size=a.shape), 1e-3)
            b2 = np.maximum(b + rng.normal(scale=0.3, size=b.shape), 1e-3)
            assert A.compute_elbo(data, a2, b2, m, v, phi, alpha, s2, tuple(mu0), tau2) <= base + 1e-9
            m2 = m + rng.normal(scale=0.05, size=m.shape)
            v2 = np.maximum(v * np.exp(rng.normal(scale=0.1, size=v.shape)), 1e-9)
            assert A.compute_elbo(data, a, b, m2, v2, phi, alpha, s2, tuple(mu0), tau2) <= base + 1e-9
        phi_new = A.update_responsibilities(data, a, b, m, v, s2)
        best = A.compute_elbo(data, a, b, m, v, phi_new, alpha, s2, tuple(mu0), tau2)
        for _ in range(15):
            ph = np.clip(phi_new + rng.normal(scale=0.05, size=phi_new.shape), 1e-6, None)
            ph /= ph.sum(1, keepdims=True)
            assert A.compute_elbo(data, a, b, m, v, ph, alpha, s2, tuple(mu0), tau2) <= best + 1e-9


def test_rand_index_matches_sklearn():
    metrics = pytest.importorskip("sklearn.metrics")
    for seed in range(5):
        instance = generate_instance(60, 4, 0.3, seed)
        labels = A.run(instance.as_array(), 0.5, 0.3, 8, seed).final_labels
        assert rand_index(instance.true_labels, labels) == pytest.approx(
            metrics.rand_score(instance.true_labels, labels), abs=1e-12
        )
