"""Zufällige 2D-Punktwolken für die DPMM-Demo: k kreisförmige, gleich gestreute
Gauß-Gruppen auf einem Ring (wie kmeans-demo) - bewusst OHNE Elongation/Kovarianz-
Extras (das ist bereits gmm-demos Thema). Die einzige Schwierigkeitsachse hier ist die
Wahl des Konzentrationsparameters α, nicht die Szenario-Geometrie selbst."""

from dataclasses import dataclass

import numpy as np

from dp_constants import RING_RADIUS


@dataclass(frozen=True)
class ClusteringInstance:
    points: tuple  # ((x, y), ...), Erzeugungsreihenfolge nach Gruppe gruppiert
    true_labels: tuple
    true_centers: tuple  # ((x, y), ...), k Einträge
    k: int

    @property
    def n_points(self):
        return len(self.points)

    def as_array(self):
        return np.array(self.points, dtype=float)


def generate_instance(n_points, k, spread, seed):
    """spread steuert die Streuung je Gruppe relativ zum Ring-Radius (wie kmeans-demo) -
    klein: Gruppen klar getrennt, groß: Gruppen überlappen sich spürbar."""
    rng = np.random.default_rng(seed)

    angles = np.linspace(0, 2 * np.pi, k, endpoint=False) + rng.uniform(-0.15, 0.15, size=k)
    centers = np.stack([RING_RADIUS * np.cos(angles), RING_RADIUS * np.sin(angles)], axis=1)
    std = max(spread, 0.05) * RING_RADIUS

    counts = np.full(k, n_points // k)
    counts[-1] += n_points - counts.sum()

    points_per_group, labels_per_group = [], []
    for i in range(k):
        pts = rng.normal(loc=centers[i], scale=std, size=(counts[i], 2))
        points_per_group.append(pts)
        labels_per_group.append(np.full(counts[i], i))

    points = np.concatenate(points_per_group, axis=0)
    labels = np.concatenate(labels_per_group, axis=0)

    return ClusteringInstance(
        points=tuple(map(tuple, points.tolist())),
        true_labels=tuple(int(l) for l in labels),
        true_centers=tuple(map(tuple, centers.tolist())),
        k=k,
    )
