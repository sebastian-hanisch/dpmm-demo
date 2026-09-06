import pytest

import dp_constants as C
from dp_algorithm import run
from dp_evaluation import alpha_comparison, rand_index
from dp_scenario import generate_instance


def test_rand_index_is_one_for_identical_partitions():
    assert rand_index([0, 0, 1, 1], [0, 0, 1, 1]) == pytest.approx(1.0)


def test_rand_index_is_one_up_to_relabeling():
    assert rand_index([0, 0, 1, 1], [5, 5, 9, 9]) == pytest.approx(1.0)


def test_rand_index_penalizes_disagreement():
    assert rand_index([0, 0, 1, 1], [0, 1, 0, 1]) < 0.6


@pytest.mark.parametrize("seed", [1, 3, 4])
def test_reasonable_alpha_recovers_true_cluster_count(seed):
    """Kern-Nachweis: bei gut getrennten Gruppen und einem moderaten alpha trifft
    Variational Inference die wahre Clusterzahl, OHNE dass sie irgendwo vorgegeben
    wurde. Parameter empirisch per Sweep-Skript verifiziert (siehe
    project_dpmm_demo_venv.md)."""
    instance = generate_instance(90, 3, 0.15, seed=seed)
    results = alpha_comparison(
        instance.as_array(), true_k=3, alpha_values=[0.3], sigma2=0.09, truncation=10, seed=1, max_iter=C.MAX_ITERATIONS
    )
    assert results[0.3]["found_k"] == 3


def test_small_alpha_causes_underclustering():
    """Bei überlappenden wahren Gruppen kann ein sehr kleines alpha genuine, aber nur
    mäßig getrennte Gruppen verschmelzen - Variational Inference bleibt dabei über
    einen weiten alpha-Bereich bemerkenswert STABIL (ein bekannter, dokumentierter
    Unterschied zu MCMC: die Mean-Field-Näherung "committet" sich meist entschlossen
    auf einen lokalen Modus, siehe project-memory), daher ist ein sehr kleines alpha
    UND ein auf die Überlappung abgestimmtes sigma nötig, um Unterclustering zu zeigen."""
    instance = generate_instance(120, 4, 0.5, seed=1)
    results = alpha_comparison(
        instance.as_array(), true_k=4, alpha_values=[0.01], sigma2=1.0, truncation=10, seed=1, max_iter=C.MAX_ITERATIONS
    )
    assert results[0.01]["found_k"] < 4


def test_large_alpha_causes_overclustering():
    instance = generate_instance(120, 4, 0.3, seed=4)
    results = alpha_comparison(
        instance.as_array(), true_k=4, alpha_values=[8.0], sigma2=0.36, truncation=10, seed=1, max_iter=C.MAX_ITERATIONS
    )
    assert results[8.0]["found_k"] > 4


def test_low_truncation_hard_caps_found_cluster_count():
    """Neuer, VI-eigener Fehlermodus: eine zu niedrige Trunkierungsgrenze T deckelt die
    gefundene Clusteranzahl hart, unabhängig von alpha - anders als beim alten
    Collapsed-Gibbs-Sampler, der (asymptotisch) beliebig viele Cluster eröffnen konnte."""
    instance = generate_instance(120, 6, 0.15, seed=1)
    result = run(instance.as_array(), alpha=1.0, sigma2=0.09, truncation=3, seed=1)
    assert result.final_step.n_active_components <= 3


def _run_as_app_would(preset_name):
    """Repliziert exakt app.py's `_compute_run`: die App nutzt den Szenario-Seed AUCH
    als Inferenz-Seed für die Primäransicht (etabliertes Muster aus gmm-demo/kmeans-demo -
    nur der Alpha-Vergleich/die Kleinmultiples nutzen einen davon entkoppelten, festen
    Seed). Presets MÜSSEN mit GENAU dieser Kopplung getestet werden, nicht mit einem
    beliebigen anderen Seed - Lehre aus der ersten Version dieses Presets (siehe
    project-memory)."""
    p = C.PRESETS[preset_name]
    instance = generate_instance(p["n_points"], p["k"], p["spread"], p["seed"])
    result = run(instance.as_array(), p["alpha"], p["sigma"] ** 2, p["truncation"], p["seed"])
    return instance, result


def test_underclustering_preset_matches_actual_app_behavior():
    instance, result = _run_as_app_would("Zu kleines α (Unterclustering)")
    assert result.final_step.n_active_components < instance.k


def test_overclustering_preset_matches_actual_app_behavior():
    instance, result = _run_as_app_would("Zu großes α (Überclustering)")
    assert result.final_step.n_active_components > instance.k


def test_simple_preset_matches_actual_app_behavior():
    instance, result = _run_as_app_would("Einfaches Beispiel")
    assert result.final_step.n_active_components == instance.k


def test_many_groups_preset_matches_actual_app_behavior():
    instance, result = _run_as_app_would("Viele Gruppen")
    assert result.final_step.n_active_components == instance.k


def test_low_truncation_preset_matches_actual_app_behavior():
    instance, result = _run_as_app_would("Trunkierung zu niedrig")
    p = C.PRESETS["Trunkierung zu niedrig"]
    assert p["truncation"] < instance.k
    assert result.final_step.n_active_components <= p["truncation"]
