"""Dirichlet-Process-Mixture-Modell (DPMM) via Collapsed Gibbs Sampling (Neal, 2000,
"Algorithm 3") - from scratch. Sphärische Komponenten mit BEKANNTER Varianz sigma2 und
einem Normal-Normal-konjugierten Prior auf die Cluster-Mittelwerte, damit sich die
Cluster-Mittelwerte analytisch herausintegrieren lassen (Collapsed Sampler) - der
Konzentrationsparameter alpha steuert über den Chinese-Restaurant-Process (CRP), wie
leicht ein neues Cluster eröffnet wird, und ersetzt k als Regler vollständig.

sklearn.mixture.BayesianGaussianMixture ist in tests/ nur ein GROBER, QUALITATIVER
Kreuzvergleich (Variational Inference statt Gibbs-Sampling - kein exaktes Zahlen-Match zu
erwarten). Die eigentliche Korrektheitsprüfung des Samplers erfolgt stattdessen exakt: auf
einer winzigen Instanz werden ALLE möglichen Partitionen enumeriert und ihre exakte
Posterior-Wahrscheinlichkeit berechnet (siehe tests/test_algorithm.py), gegen die die
empirische Verteilung vieler unabhängiger Sampler-Läufe verglichen wird."""

from dataclasses import dataclass

import numpy as np

from dp_constants import MAX_SWEEPS, PRIOR_MEAN, PRIOR_VARIANCE


@dataclass(frozen=True)
class Sweep:
    sweep: int  # 0 = Startzuordnung (jeder Punkt sein eigenes Cluster), danach je ein voller Sweep
    labels: tuple  # fortlaufend 0..k-1 nummeriert (Erzeugungsreihenfolge)
    n_clusters: int


@dataclass(frozen=True)
class RunResult:
    sweeps: tuple  # Sweep-Folge in Ausführungsreihenfolge
    alpha: float
    sigma2: float

    @property
    def final_sweep(self):
        return self.sweeps[-1]

    @property
    def final_labels(self):
        return self.final_sweep.labels

    @property
    def cluster_count_trace(self):
        return tuple(s.n_clusters for s in self.sweeps)


def _relabel(labels):
    """Fortlaufend 0..k-1 nummerieren, in Erzeugungsreihenfolge - wie die übrigen Demos
    dieser Reihe (kmeans-demo, agglomerative-demo, ...)."""
    seen = {}
    result = []
    for l in labels:
        if l not in seen:
            seen[l] = len(seen)
        result.append(seen[l])
    return tuple(result)


def _predictive_params(member_sum, n_members, sigma2, mu0, tau2):
    """Posterior-Prädiktivverteilung für einen NEUEN Punkt, gegeben die Summe von
    `n_members` bereits einem Cluster zugeordneten Punkten - Normal-Normal-Konjugation,
    das Cluster-Mittel ist analytisch herausintegriert. Für n_members=0 liefert das
    exakt die Prior-Prädiktivverteilung (ein neues, leeres Cluster)."""
    mu0 = np.asarray(mu0, dtype=float)
    post_var = 1.0 / (1.0 / tau2 + n_members / sigma2)
    post_mean = post_var * (mu0 / tau2 + np.asarray(member_sum, dtype=float) / sigma2)
    predictive_var = post_var + sigma2
    return post_mean, predictive_var


def _log_gaussian_isotropic(x, mean, var):
    """Log-Dichte einer isotropen (sphärischen) multivariaten Normalverteilung."""
    diff = np.asarray(x, dtype=float) - mean
    d = len(mean)
    return -0.5 * (d * np.log(2 * np.pi * var) + float(diff @ diff) / var)


def crp_log_prior_existing(n_c, n_minus_1, alpha):
    """CRP-Prior (log): Punkt schließt sich einem bestehenden Cluster mit `n_c` ANDEREN
    Mitgliedern an - Wahrscheinlichkeit proportional zur Clustergröße ("rich get
    richer"). `n_minus_1` = n-1, der für alle Optionen gemeinsame Nenner."""
    return np.log(n_c) - np.log(n_minus_1 + alpha)


def crp_log_prior_new(n_minus_1, alpha):
    """CRP-Prior (log): Punkt eröffnet ein neues Cluster - Wahrscheinlichkeit
    proportional zum Konzentrationsparameter alpha."""
    return np.log(alpha) - np.log(n_minus_1 + alpha)


def conditional_distribution(x_i, counts, sums, alpha, sigma2, mu0, tau2, n_minus_1):
    """Die volle bedingte Verteilung für EINEN Punkt (Algorithm 3, Neal 2000): je
    bestehendem Cluster (aus `counts`/`sums`, OHNE den betrachteten Punkt) die CRP-
    Prior-Wahrscheinlichkeit mal Posterior-Prädiktiv-Likelihood, plus die Option "neues
    Cluster". Gibt (candidates, probs) zurück, mit `probs` normiert (Summe 1) - als
    eigene, direkt testbare Funktion ausgelagert (siehe tests/test_algorithm.py)."""
    candidates, log_weights = [], []
    for c, n_c in counts.items():
        mean, var = _predictive_params(sums[c], n_c, sigma2, mu0, tau2)
        log_w = crp_log_prior_existing(n_c, n_minus_1, alpha) + _log_gaussian_isotropic(x_i, mean, var)
        candidates.append(c)
        log_weights.append(log_w)

    new_label = (max(counts.keys()) + 1) if counts else 0
    mean_new, var_new = _predictive_params(np.zeros(2), 0, sigma2, mu0, tau2)
    log_w_new = crp_log_prior_new(n_minus_1, alpha) + _log_gaussian_isotropic(x_i, mean_new, var_new)
    candidates.append(new_label)
    log_weights.append(log_w_new)

    log_weights = np.array(log_weights)
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    probs = weights / weights.sum()
    return candidates, probs


def gibbs_sweep(data, labels, alpha, sigma2, mu0, tau2, rng):
    """Ein vollständiger Sweep: jeder Punkt (in fester Reihenfolge 0..n-1) bekommt einmal
    ein neu gezogenes Label, nach Algorithm 3 (Neal, 2000). Effiziente inkrementelle
    Buchführung über Cluster-Größe/-Summe (O(1) Entfernen/Einfügen) statt bei jedem Punkt
    alle übrigen n-1 Punkte neu zu durchsuchen - relevant, weil die Startzuordnung (jeder
    Punkt sein eigenes Cluster) im ersten Sweep bis zu n aktive Cluster gleichzeitig hat.
    Modifiziert `labels` (Liste) direkt und gibt sie zurück."""
    n = len(data)
    counts, sums = {}, {}
    for i, l in enumerate(labels):
        counts[l] = counts.get(l, 0) + 1
        sums[l] = sums.get(l, np.zeros(2)) + data[i]

    for i in range(n):
        x_i = data[i]
        current = labels[i]
        counts[current] -= 1
        sums[current] = sums[current] - x_i
        if counts[current] == 0:
            del counts[current]
            del sums[current]

        candidates, probs = conditional_distribution(x_i, counts, sums, alpha, sigma2, mu0, tau2, n - 1)
        choice = candidates[rng.choice(len(candidates), p=probs)]

        labels[i] = choice
        counts[choice] = counts.get(choice, 0) + 1
        sums[choice] = sums.get(choice, np.zeros(2)) + x_i

    return labels


def run(data, alpha, sigma2, seed, n_sweeps=MAX_SWEEPS, mu0=PRIOR_MEAN, tau2=PRIOR_VARIANCE):
    """Führt den Collapsed-Gibbs-Sampler vollständig protokolliert aus: Schritt 0 ist die
    Startzuordnung (jeder Punkt sein eigenes Cluster - eine neutrale, informationsfreie
    Initialisierung, die weder Über- noch Unterclustering vorwegnimmt), jeder weitere
    Schritt ein vollständiger Sweep. Anders als k-Means' Fixpunkt oder EMs Toleranzschwelle
    gibt es HIER kein Konvergenzkriterium im klassischen Sinn - MCMC "konvergiert" zu
    einer stationären VERTEILUNG über Partitionen, nicht zu einem festen Zustand, daher
    läuft der Sampler immer genau `n_sweeps` Schritte."""
    rng = np.random.default_rng(seed)
    data = np.asarray(data, dtype=float)
    n = len(data)

    labels = list(range(n))
    sweeps = [Sweep(sweep=0, labels=_relabel(labels), n_clusters=len(set(labels)))]

    for sweep_idx in range(1, n_sweeps + 1):
        labels = gibbs_sweep(data, labels, alpha, sigma2, mu0, tau2, rng)
        relabeled = _relabel(labels)
        sweeps.append(Sweep(sweep=sweep_idx, labels=relabeled, n_clusters=len(set(relabeled))))

    return RunResult(sweeps=tuple(sweeps), alpha=alpha, sigma2=sigma2)
