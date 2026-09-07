"""Zufällige 2D-Punktwolken für die DPMM-Demo: k kreisförmige, gleich gestreute
Gauß-Gruppen auf einem Ring ("blobs", wie kmeans-demo) oder k nicht-konvexe
Halbkreis-Bögen ("moons", generalisiert wie in dbscan-demo/kmeans-demo) - bewusst OHNE
Elongation/Kovarianz-Extras (das ist bereits gmm-demos Thema) und ohne
Größen-/Dichte-Ungleichgewicht. Die primäre Schwierigkeitsachse bleibt die Wahl des
Konzentrationsparameters α, nicht die Szenario-Geometrie - shape ist eine zusätzliche,
unabhängige Frage ("meistert DPMM auch nicht-konvexe Formen?"), keine neue
Ungleichgewichts-Achse."""

from dataclasses import dataclass

import numpy as np

from dp_constants import ARC_RADIUS, ARC_RING_RADIUS, RING_RADIUS


@dataclass(frozen=True)
class ClusteringInstance:
    points: tuple  # ((x, y), ...), Erzeugungsreihenfolge nach Gruppe gruppiert
    true_labels: tuple
    true_centers: tuple  # ((x, y), ...), k Einträge - bei "moons" der Schwerpunkt je Bogen
    shape: str  # "blobs" oder "moons"
    k: int

    @property
    def n_points(self):
        return len(self.points)

    def as_array(self):
        return np.array(self.points, dtype=float)


def _generate_blobs(n_points, k, spread, rng):
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
    return points, labels, centers


def _generate_moons(n_points, k, spread, rng):
    """k=2: das klassische "two moons"-Beispiel (wie in dbscan-demo/kmeans-demo). k>2: k
    Halbkreis-Bögen wie Blütenblätter auf einem Ring, konkave Seite zum Zentrum. Keine
    Größen-/Dichte-Ungleichgewichts-Achse hier, wie bei "blobs": alle Gruppen bekommen
    gleich viele Punkte und dasselbe Rauschen."""
    counts = np.full(k, n_points // k)
    counts[-1] += n_points - counts.sum()
    noise_std = max(spread, 0.05) * ARC_RADIUS * 0.3

    if k == 2:
        t1 = rng.uniform(0, np.pi, counts[0])
        x1 = ARC_RADIUS * np.cos(t1)
        y1 = ARC_RADIUS * np.sin(t1)

        t2 = rng.uniform(0, np.pi, counts[1])
        x2 = ARC_RADIUS * (1 - np.cos(t2))
        y2 = ARC_RADIUS * (0.5 - np.sin(t2))

        pts1 = np.stack([x1, y1], axis=1) + rng.normal(scale=noise_std, size=(counts[0], 2))
        pts2 = np.stack([x2, y2], axis=1) + rng.normal(scale=noise_std, size=(counts[1], 2))
        points = np.concatenate([pts1, pts2], axis=0)
        labels = np.concatenate([np.zeros(counts[0], dtype=int), np.ones(counts[1], dtype=int)])
        centers = np.stack([points[labels == i].mean(axis=0) for i in range(k)])
        return points, labels, centers

    layout_angles = np.linspace(0, 2 * np.pi, k, endpoint=False) + rng.uniform(-0.1, 0.1, size=k)
    arc_centers = np.stack(
        [ARC_RING_RADIUS * np.cos(layout_angles), ARC_RING_RADIUS * np.sin(layout_angles)], axis=1
    )

    points_per_group, labels_per_group = [], []
    for i in range(k):
        t = rng.uniform(0, np.pi, counts[i])
        local_x = ARC_RADIUS * np.cos(t)
        local_y = ARC_RADIUS * np.sin(t)
        # Um layout_angle_i + pi rotieren, damit die konkave Seite des Bogens zum
        # Ringzentrum zeigt (Blütenblatt-Anordnung), statt nach außen.
        rot = layout_angles[i] + np.pi
        cos_r, sin_r = np.cos(rot), np.sin(rot)
        rx = cos_r * local_x - sin_r * local_y
        ry = sin_r * local_x + cos_r * local_y
        pts = np.stack([rx, ry], axis=1) + arc_centers[i] + rng.normal(scale=noise_std, size=(counts[i], 2))
        points_per_group.append(pts)
        labels_per_group.append(np.full(counts[i], i))

    points = np.concatenate(points_per_group, axis=0)
    labels = np.concatenate(labels_per_group, axis=0)
    centers = np.stack([points[labels == i].mean(axis=0) for i in range(k)])
    return points, labels, centers


def generate_instance(n_points, k, spread, seed, shape="blobs"):
    """spread steuert je nach shape entweder die Streuung je Gruppe relativ zum
    Ring-Radius ("blobs") oder das Rauschen um die ideale Bogen-Kurve ("moons") - klein:
    Gruppen klar getrennt, groß: Gruppen überlappen sich spürbar."""
    rng = np.random.default_rng(seed)

    if shape == "moons":
        points, labels, centers = _generate_moons(n_points, k, spread, rng)
    else:
        points, labels, centers = _generate_blobs(n_points, k, spread, rng)

    return ClusteringInstance(
        points=tuple(map(tuple, points.tolist())),
        true_labels=tuple(int(l) for l in labels),
        true_centers=tuple(map(tuple, centers.tolist())),
        shape=shape,
        k=k,
    )
