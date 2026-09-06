from dp_scenario import generate_instance


def test_reproducible_with_same_seed():
    a = generate_instance(100, 3, 0.2, seed=7)
    b = generate_instance(100, 3, 0.2, seed=7)
    assert a.points == b.points
    assert a.true_labels == b.true_labels


def test_different_seed_gives_different_points():
    a = generate_instance(100, 3, 0.2, seed=1)
    b = generate_instance(100, 3, 0.2, seed=2)
    assert a.points != b.points


def test_n_points_and_k_respected():
    instance = generate_instance(97, 4, 0.2, seed=5)
    assert instance.n_points == 97
    assert instance.k == 4
    assert set(instance.true_labels) == {0, 1, 2, 3}


def test_true_centers_are_well_separated_relative_to_spread():
    instance = generate_instance(90, 3, 0.15, seed=1)
    centers = instance.as_array()[[instance.true_labels.index(i) for i in range(3)]]
    # grobe Sanity-Pruefung ueber die tatsaechlichen Gruppenmittelwerte, nicht die
    # (verrauschten) true_centers direkt
    import numpy as np

    points = instance.as_array()
    labels = np.array(instance.true_labels)
    means = np.array([points[labels == i].mean(axis=0) for i in range(3)])
    dists = [
        np.linalg.norm(means[i] - means[j])
        for i in range(3)
        for j in range(i + 1, 3)
    ]
    assert min(dists) > 1.0
