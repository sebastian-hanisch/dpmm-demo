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


@pytest.mark.parametrize("seed", [1, 5, 6])
def test_reasonable_alpha_recovers_true_cluster_count(seed):
    """Kern-Nachweis: bei gut getrennten Gruppen (spread klein relativ zu sigma) und
    einem moderaten alpha trifft der Sampler die wahre Clusterzahl, OHNE dass sie
    irgendwo vorgegeben wurde. Parameter empirisch per Sweep-Skript verifiziert (siehe
    project_dpmm_demo_venv.md) - bei diesem sigma2 trifft alpha=0.3 die wahre
    Clusterzahl deutlich zuverlässiger als alpha=1.0."""
    instance = generate_instance(90, 3, 0.15, seed=seed)
    results = alpha_comparison(instance.as_array(), true_k=3, alpha_values=[0.3], sigma2=0.09, seed=1, n_sweeps=30)
    assert results[0.3]["found_k"] == 3


def test_small_alpha_causes_underclustering():
    """Bei STARK überlappenden wahren Gruppen (spread=0.9, sigma darauf abgestimmt)
    kann ein sehr kleines alpha genuine, aber nur mäßig getrennte Gruppen verschmelzen -
    bei klar getrennten Gruppen (wie im 'Einfaches Beispiel'-Preset) reicht dafür auch
    ein beliebig kleines alpha nicht aus (der CRP-Prior-Gewinn ist dann kleiner als der
    Likelihood-Verlust einer Fehlzuordnung), was selbst ein reales Merkmal des
    Ein-Punkt-Gibbs-Samplers ist, nicht ein Test-Artefakt - siehe project-memory."""
    instance = generate_instance(120, 4, 0.9, seed=2)
    results = alpha_comparison(
        instance.as_array(), true_k=4, alpha_values=[0.01], sigma2=3.24, seed=1, n_sweeps=40
    )
    assert results[0.01]["found_k"] < 4


def test_large_alpha_causes_overclustering():
    instance = generate_instance(120, 4, 0.9, seed=2)
    results = alpha_comparison(
        instance.as_array(), true_k=4, alpha_values=[0.5], sigma2=3.24, seed=1, n_sweeps=40
    )
    assert results[0.5]["found_k"] > 4


def _run_as_app_would(preset_name):
    """Repliziert exakt app.py's `_compute_run`: die App nutzt den Szenario-Seed AUCH
    als Sampler-Seed für die Primäransicht (etabliertes Muster aus gmm-demo/kmeans-demo -
    nur der Alpha-Vergleich/die Kleinmultiples nutzen einen davon entkoppelten, festen
    Sampler-Seed). Presets MÜSSEN mit GENAU dieser Kopplung getestet werden, nicht mit
    einem beliebigen anderen Sampler-Seed - sonst kann ein Preset in `alpha_comparison`-
    Tests funktionieren, aber live in der App ein anderes (MCMC-bedingt durchaus
    plausibles) Ergebnis zeigen, wie bei der ersten Version dieses Presets geschehen."""
    p = C.PRESETS[preset_name]
    instance = generate_instance(p["n_points"], p["k"], p["spread"], p["seed"])
    result = run(instance.as_array(), p["alpha"], p["sigma"] ** 2, p["seed"], n_sweeps=C.MAX_SWEEPS)
    return instance, result


def test_underclustering_preset_matches_actual_app_behavior():
    instance, result = _run_as_app_would("Zu kleines α (Unterclustering)")
    assert result.final_sweep.n_clusters < instance.k


def test_overclustering_preset_matches_actual_app_behavior():
    instance, result = _run_as_app_would("Zu großes α (Überclustering)")
    assert result.final_sweep.n_clusters > instance.k


def test_simple_preset_matches_actual_app_behavior():
    instance, result = _run_as_app_would("Einfaches Beispiel")
    assert result.final_sweep.n_clusters == instance.k


def test_many_groups_preset_matches_actual_app_behavior():
    instance, result = _run_as_app_would("Viele Gruppen")
    assert result.final_sweep.n_clusters == instance.k


def test_alpha_sensitivity_is_monotonic_on_average():
    """Größeres alpha soll (auf demselben Datensatz) nicht zu WENIGER Clustern führen
    als ein kleineres alpha - die grundlegende CRP-Monotonie-Intuition."""
    instance = generate_instance(120, 4, 0.9, seed=2)
    results = alpha_comparison(
        instance.as_array(), true_k=4, alpha_values=[0.01, 0.1, 0.5], sigma2=3.24, seed=1, n_sweeps=40
    )
    found = [results[a]["found_k"] for a in [0.01, 0.1, 0.5]]
    assert found[0] <= found[1] <= found[2]
