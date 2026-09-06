"""Dirichlet-Process-Mixture-Modell (DPMM) via Mean-Field Variational Inference (Blei &
Jordan, 2006) - from scratch. Gleiches Modell wie zuvor: sphärische Komponenten mit
BEKANNTER Varianz sigma2 und einem Normal-Normal-konjugierten Prior auf die
Cluster-Mittelwerte - aber statt eines Collapsed-Gibbs-Samplers (MCMC, asymptotisch
exakt, aber langsam und ohne festen Endzustand) wird hier eine TRUNKIERTE
Stick-Breaking-Repräsentation per Koordinatenaufstieg optimiert: schneller, deterministisch
(bei fester Initialisierung), mit einem echten Konvergenzkriterium (ELBO-Toleranz) - wie
gmm-demos EM, nur mit einer zusätzlichen Stick-Breaking-Dimension statt fester Gewichte.

Ehrlicher neuer Preis: die Trunkierungsgrenze T ersetzt "unendlich viele mögliche
Cluster" durch eine vorab festgelegte Obergrenze.

sklearn.mixture.BayesianGaussianMixture ist in tests/ weiterhin nur ein GROBER,
QUALITATIVER Kreuzvergleich (volle statt sphärischer/bekannter Kovarianz - kein exaktes
Zahlen-Match zu erwarten), methodisch aber treffender als zuvor: beide sind jetzt
tatsächlich Variational Inference für eine DP-Mixture. Die exakte Korrektheitsprüfung
bleibt unabhängig von jeder Inferenzmethode: auf einer winzigen Instanz wird die exakte
MAP-Partition durch Enumeration aller Partitionen berechnet (siehe tests/test_algorithm.py) -
konvergiert Variational Inference zu derselben MAP-Partition?"""

from dataclasses import dataclass

import numpy as np
from scipy.special import digamma, gammaln

from dp_constants import ELBO_TOL, MAX_ITERATIONS, PRIOR_MEAN, PRIOR_VARIANCE


@dataclass(frozen=True)
class Step:
    iteration: int  # 0 = Responsibilities auf den Startparametern, danach je ein voller Koordinatenaufstiegs-Zyklus
    stick_a: tuple  # (T-1,) Beta-Parameter gamma_{t,1}
    stick_b: tuple  # (T-1,) Beta-Parameter gamma_{t,2}
    means: tuple  # (T, 2) Variations-Mittelwerte m_t
    responsibilities: tuple  # (n, T)
    elbo: float
    n_changed: int  # Punkte, deren hartes Label (argmax responsibility) sich gegenüber dem Vorschritt änderte

    @property
    def hard_labels(self):
        resp = np.array(self.responsibilities)
        return tuple(int(l) for l in resp.argmax(axis=1))

    @property
    def n_active_components(self):
        return len(set(self.hard_labels))


@dataclass(frozen=True)
class RunResult:
    steps: tuple  # Step-Folge in Ausführungsreihenfolge
    alpha: float
    sigma2: float
    truncation: int
    converged: bool  # True, wenn die relative ELBO-Verbesserung unter die Toleranz fiel
    truncated: bool  # True, wenn max_iter erreicht wurde, ohne dass die Toleranz erreicht wurde

    @property
    def final_step(self):
        return self.steps[-1]

    @property
    def final_labels(self):
        return self.final_step.hard_labels

    @property
    def cluster_count_trace(self):
        return tuple(s.n_active_components for s in self.steps)


def _expected_log_stick_weights(stick_a, stick_b):
    """E[log pi_t] für t=1..T unter q(beta_t)=Beta(gamma_t1,gamma_t2), t=1..T-1, mit dem
    letzten Stück beta_T=1 (Trunkierung). E[log beta_t] und E[log(1-beta_t)] über die
    Digamma-Funktion (Standardresultat für die Beta-Verteilung)."""
    digamma_sum = digamma(stick_a + stick_b)
    e_log_beta = digamma(stick_a) - digamma_sum
    e_log_1m_beta = digamma(stick_b) - digamma_sum

    cumulative = np.concatenate([[0.0], np.cumsum(e_log_1m_beta)])
    e_log_pi_existing = e_log_beta + cumulative[:-1]
    e_log_pi_last = cumulative[-1]  # beta_T=1 -> log(beta_T)=0, nur das Produkt der vorherigen (1-beta_s)
    return np.concatenate([e_log_pi_existing, [e_log_pi_last]])


def _expected_log_gaussian(data, mean, comp_var, sigma2, d=2):
    """E_q(mu)[log N(x|mu,sigma2 I)] mit mu~N(mean,comp_var I): E[||x-mu||^2] =
    ||x-mean||^2 + d*comp_var (Standardresultat für die quadratische Form unter einer
    Normalverteilung), sigma2 ist die BEKANNTE Beobachtungsvarianz des Modells."""
    diff_sq = ((data - mean) ** 2).sum(axis=1)
    return -0.5 * d * np.log(2 * np.pi * sigma2) - 0.5 * (diff_sq + d * comp_var) / sigma2


def update_responsibilities(data, stick_a, stick_b, means, comp_var, sigma2):
    """E-Schritt-Analogon: Responsibilities via Softmax im Log-Raum (Log-Sum-Exp-Trick)
    aus E[log pi_t] + E_q(mu_t)[log N(x|mu_t,sigma2)]."""
    n, d = data.shape
    T = len(means)
    e_log_pi = _expected_log_stick_weights(stick_a, stick_b)

    log_weighted = np.empty((n, T))
    for t in range(T):
        e_log_lik = _expected_log_gaussian(data, means[t], comp_var[t], sigma2, d)
        log_weighted[:, t] = e_log_pi[t] + e_log_lik

    max_log = log_weighted.max(axis=1, keepdims=True)
    log_sum = max_log[:, 0] + np.log(np.exp(log_weighted - max_log).sum(axis=1))
    responsibilities = np.exp(log_weighted - log_sum[:, None])
    return responsibilities


def update_stick_breaking(responsibilities, alpha):
    """M-Schritt-Analogon fuer die Stick-Breaking-Gewichte: gamma_{t,1}=1+N_t,
    gamma_{t,2}=alpha+sum_{s>t}N_s, mit N_t = sum_i phi_{i,t} - nur fuer t=1..T-1 (das
    letzte Stueck beta_T=1 ist per Trunkierung fixiert, kein Update noetig)."""
    n_t = responsibilities.sum(axis=0)
    T = len(n_t)
    tail_sum = np.array([n_t[t + 1 :].sum() for t in range(T - 1)])
    stick_a = 1.0 + n_t[:-1]
    stick_b = alpha + tail_sum
    return stick_a, stick_b


def update_components(data, responsibilities, sigma2, mu0, tau2):
    """M-Schritt-Analogon fuer die Komponenten-Mittelwerte: Normal-Normal-Update mit den
    erwarteten (responsibility-gewichteten) Suffizienzstatistiken - derselbe
    Normal-Normal-Mechanismus wie im alten Collapsed-Sampler, hier aber auf WEICHEN
    (responsibility-gewichteten) statt harten Cluster-Zuordnungen."""
    mu0 = np.asarray(mu0, dtype=float)
    n_t = responsibilities.sum(axis=0)
    weighted_sum = responsibilities.T @ data  # (T, 2)

    comp_var = 1.0 / (1.0 / tau2 + n_t / sigma2)
    means = comp_var[:, None] * (mu0[None, :] / tau2 + weighted_sum / sigma2)
    return means, comp_var


def _kl_beta(a1, b1, a2, b2):
    """KL(Beta(a1,b1) || Beta(a2,b2)), geschlossene Form."""
    log_beta1 = gammaln(a1) + gammaln(b1) - gammaln(a1 + b1)
    log_beta2 = gammaln(a2) + gammaln(b2) - gammaln(a2 + b2)
    return (
        log_beta2 - log_beta1
        + (a1 - a2) * digamma(a1)
        + (b1 - b2) * digamma(b1)
        - (a1 + b1 - a2 - b2) * digamma(a1 + b1)
    )


def _kl_normal_isotropic(m1, s1_sq, m0, tau2, d=2):
    """KL(N(m1, s1_sq*I_d) || N(m0, tau2*I_d)) fuer isotrope Normalverteilungen."""
    m1, m0 = np.asarray(m1, dtype=float), np.asarray(m0, dtype=float)
    sq_dist = float(((m1 - m0) ** 2).sum())
    return 0.5 * (d * np.log(tau2 / s1_sq) + d * s1_sq / tau2 + sq_dist / tau2 - d)


def compute_elbo(data, stick_a, stick_b, means, comp_var, responsibilities, alpha, sigma2, mu0, tau2):
    """Variationale untere Schranke (ELBO): erwartete Log-Likelihood + erwarteter
    Stick-Breaking-Log-Prior - Responsibility-Entropie, minus die KL-Terme fuer die
    Stick-Breaking- und Komponenten-Verteilungen. Koordinatenaufstieg garantiert, dass
    die ELBO bei jedem Teilschritt nie sinkt (siehe tests/test_algorithm.py)."""
    n, d = data.shape
    T = len(means)
    e_log_pi = _expected_log_stick_weights(stick_a, stick_b)

    expected_log_lik = 0.0
    for t in range(T):
        e_log_lik_t = _expected_log_gaussian(data, means[t], comp_var[t], sigma2, d)
        expected_log_lik += float((responsibilities[:, t] * e_log_lik_t).sum())

    expected_log_prior_z = float((responsibilities * e_log_pi[None, :]).sum())

    safe_resp = np.clip(responsibilities, 1e-300, 1.0)
    entropy_z = -float((responsibilities * np.log(safe_resp)).sum())

    kl_stick = sum(_kl_beta(stick_a[t], stick_b[t], 1.0, alpha) for t in range(T - 1))
    kl_components = sum(_kl_normal_isotropic(means[t], comp_var[t], mu0, tau2, d) for t in range(T))

    return expected_log_lik + expected_log_prior_z + entropy_z - kl_stick - kl_components


def init_params(data, truncation, mu0, tau2, rng):
    """T verschiedene Datenpunkte als Startmittelwerte (wie gmm-demos `init_params`),
    Start-Varianz = Prior-Varianz (noch keine Information aus den Daten), Start-Stick-
    Breaking-Parameter = Prior-Erwartung (gamma_1=1, gamma_2=alpha wird beim ersten aufruf
    von `update_stick_breaking` ohnehin sofort ueberschrieben - hier nur ein neutraler
    Platzhalter fuer den allerersten Responsibility-Schritt)."""
    n = len(data)
    idx = rng.choice(n, size=min(truncation, n), replace=False)
    means = data[idx].copy()
    if len(means) < truncation:
        pad = np.tile(means[-1], (truncation - len(means), 1))
        means = np.concatenate([means, pad], axis=0)
    comp_var = np.full(truncation, tau2)
    stick_a = np.ones(truncation - 1)
    stick_b = np.ones(truncation - 1)
    return stick_a, stick_b, means, comp_var


def run(data, alpha, sigma2, truncation, seed, max_iter=MAX_ITERATIONS, tol=ELBO_TOL, mu0=PRIOR_MEAN, tau2=PRIOR_VARIANCE):
    """Fuehrt Variational Inference vollstaendig protokolliert aus: Schritt 0 sind die
    Responsibilities auf den Startparametern, jeder weitere Schritt ein vollstaendiger
    Stick-Breaking-dann-Komponenten-dann-Responsibilities-Zyklus. Terminiert, sobald die
    relative ELBO-Verbesserung unter `tol` faellt (Koordinatenaufstieg ist nachweisbar
    ELBO-monoton, siehe Mathe-Abschnitt) - ein ECHTES Konvergenzkriterium, anders als beim
    vorherigen MCMC-Sampler, der zu einer stationaeren VERTEILUNG statt einem Fixpunkt
    konvergierte."""
    rng = np.random.default_rng(seed)
    data = np.asarray(data, dtype=float)
    truncation = min(truncation, len(data))

    stick_a, stick_b, means, comp_var = init_params(data, truncation, mu0, tau2, rng)
    responsibilities = update_responsibilities(data, stick_a, stick_b, means, comp_var, sigma2)
    elbo = compute_elbo(data, stick_a, stick_b, means, comp_var, responsibilities, alpha, sigma2, mu0, tau2)
    hard_labels = responsibilities.argmax(axis=1)

    steps = [
        Step(
            iteration=0,
            stick_a=tuple(float(a) for a in stick_a),
            stick_b=tuple(float(b) for b in stick_b),
            means=tuple(map(tuple, means.tolist())),
            responsibilities=tuple(map(tuple, responsibilities.tolist())),
            elbo=elbo,
            n_changed=len(data),
        )
    ]

    converged = False
    prev_elbo = elbo
    for iteration in range(1, max_iter + 1):
        stick_a, stick_b = update_stick_breaking(responsibilities, alpha)
        means, comp_var = update_components(data, responsibilities, sigma2, mu0, tau2)
        responsibilities = update_responsibilities(data, stick_a, stick_b, means, comp_var, sigma2)
        elbo = compute_elbo(data, stick_a, stick_b, means, comp_var, responsibilities, alpha, sigma2, mu0, tau2)

        new_hard_labels = responsibilities.argmax(axis=1)
        n_changed = int(np.sum(new_hard_labels != hard_labels))
        steps.append(
            Step(
                iteration=iteration,
                stick_a=tuple(float(a) for a in stick_a),
                stick_b=tuple(float(b) for b in stick_b),
                means=tuple(map(tuple, means.tolist())),
                responsibilities=tuple(map(tuple, responsibilities.tolist())),
                elbo=elbo,
                n_changed=n_changed,
            )
        )
        hard_labels = new_hard_labels

        improvement = abs(elbo - prev_elbo) / max(abs(prev_elbo), 1e-12)
        prev_elbo = elbo
        if improvement < tol:
            converged = True
            break

    truncated = not converged
    return RunResult(
        steps=tuple(steps), alpha=alpha, sigma2=sigma2, truncation=truncation,
        converged=converged, truncated=truncated,
    )
