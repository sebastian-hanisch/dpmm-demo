"""Defaults, Regler-Grenzen, Sicherheitsgrenzen und Presets für die DPMM-Demo."""

DEFAULT_N_POINTS = 90
DEFAULT_K = 3
DEFAULT_SPREAD = 0.15
DEFAULT_ALPHA = 0.3
DEFAULT_SIGMA = 0.3
DEFAULT_TRUNCATION = 10
DEFAULT_SEED = 1
DEFAULT_SHAPE = "blobs"

N_POINTS_MIN, N_POINTS_MAX = 30, 200
K_MIN, K_MAX = 2, 6
SPREAD_MIN, SPREAD_MAX = 0.05, 0.9
ALPHA_MIN, ALPHA_MAX = 0.005, 10.0
SIGMA_MIN, SIGMA_MAX = 0.1, 2.0
TRUNCATION_MIN, TRUNCATION_MAX = 3, 15

SHAPES = ("blobs", "moons")
SHAPE_LABELS = {"blobs": "Gruppen (Blobs)", "moons": "Halbmonde"}

# Variational Inference konvergiert (anders als MCMC, das zu einer stationären
# VERTEILUNG statt einem Fixpunkt konvergiert) zu einem ECHTEN Fixpunkt - ein
# Toleranzkriterium auf die ELBO-Verbesserung, analog zu gmm-demos
# LOG_LIKELIHOOD_TOL. MAX_ITERATIONS ist nur die Sicherheitsgrenze, falls die
# Toleranz nie erreicht wird.
MAX_ITERATIONS = 100
ELBO_TOL = 1e-4

RING_RADIUS = 2.0
ARC_RADIUS = 1.7
ARC_RING_RADIUS = 4.5

# Vager, nicht als Regler exponierter Prior auf die Cluster-Mittelwerte - zentriert
# auf das Ring-Layout, mit einer Varianz, die deutlich größer als die Punktstreuung
# selbst ist (dominiert die Posterior nicht). Interne Konstante, analog zu HDBSCANs
# REG_COVAR.
PRIOR_MEAN = (0.0, 0.0)
PRIOR_VARIANCE = (2.0 * RING_RADIUS) ** 2

# Feste Referenz-Alpha-Werte für die front-and-center Alpha-Vergleichsgrafik - deckt die
# gesamte Spanne von "sehr klein" bis "sehr groß" ab, unabhängig vom aktuell eingestellten
# alpha (das zusätzlich immer mit eingemischt wird, siehe app.py).
ALPHA_SWEEP_VALUES = (0.01, 0.05, 0.2, 0.5, 1.0, 3.0, 8.0)

# Fester Sampler-Seed für den Alpha-Vergleich (Kleinmultiples + Balkendiagramm) -
# bewusst UNABHÄNGIG vom Szenario-Seed des Reglers, damit alle alpha-Werte von einer
# äquivalenten Startkonfiguration aus verglichen werden. Lehre aus gmm-demos
# COMPARISON_EM_SEED-Bug: den Szenario-Seed für die Sampler-eigene Zufälligkeit
# wiederzuverwenden koppelt beide unbeabsichtigt und macht den Vergleich instabil.
COMPARISON_SAMPLER_SEED = 1

PRESETS = {
    "Einfaches Beispiel": {
        "n_points": 90, "k": 3, "spread": 0.15, "alpha": 0.3, "sigma": 0.3, "truncation": 10,
        "shape": "blobs", "seed": 1,
    },
    "Zu kleines α (Unterclustering)": {
        "n_points": 120, "k": 4, "spread": 0.5, "alpha": 0.01, "sigma": 1.0, "truncation": 10,
        "shape": "blobs", "seed": 1,
    },
    "Zu großes α (Überclustering)": {
        "n_points": 120, "k": 4, "spread": 0.3, "alpha": 8.0, "sigma": 0.6, "truncation": 10,
        "shape": "blobs", "seed": 4,
    },
    "Trunkierung zu niedrig": {
        "n_points": 120, "k": 6, "spread": 0.15, "alpha": 1.0, "sigma": 0.3, "truncation": 3,
        "shape": "blobs", "seed": 1,
    },
    "Viele Gruppen": {
        "n_points": 180, "k": 6, "spread": 0.2, "alpha": 0.5, "sigma": 0.4, "truncation": 10,
        "shape": "blobs", "seed": 1,
    },
    "Nicht-konvexe Formen (DPMM überclustert)": {
        "n_points": 150, "k": 2, "spread": 0.1, "alpha": 0.3, "sigma": 0.3, "truncation": 10,
        "shape": "moons", "seed": 1,
    },
}
