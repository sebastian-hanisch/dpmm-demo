# Dirichlet-Process-Mixture für Sammel-Routen ohne feste Anzahl – Streamlit-Demo

Siebtes Stück der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations
Research und Machine Learning", **Fortsetzung von [gmm-demo](../gmm-demo)**: gmm-demo
benennt im Mathe-Abschnitt offen die eigene verbleibende Schwäche - die Komponentenzahl k
muss weiterhin vorab feststehen, genau wie bei k-Means. Diese Demo behebt **genau das**,
mit einem völlig anderen Mechanismus als DBSCAN/HDBSCAN (Dichte): einem
**Dirichlet-Process-Mixture-Modell (DPMM)**, dem bayesianischen, nichtparametrischen
Gegenstück. Damit schließt die Clustering-Linie eine schöne Symmetrie ab - beide Äste
enden darin, "k nicht vorab festlegen zu müssen", aber über zwei grundverschiedene Ideen:

```
kmeans-demo → dbscan-demo ──┐
                             ├──> hdbscan-demo   (kein k: über Dichte-Hierarchie)
              agglomerative-demo ──────────┘
kmeans-demo → gmm-demo → dpmm-demo             (kein k: über bayesianische Nichtparametrik)
```

Statt k geben Sie nur einen **Konzentrationsparameter α** vor, der beschreibt, wie leicht
ein neues Cluster eröffnet wird - berechnet über einen von Grund auf implementierten
**Collapsed-Gibbs-Sampler** (Neal, 2000). Ein für diese Reihe komplett neues Paradigma:
bisher exakte Suche (branch-bound-demo), lokale Suche/EM (kmeans-demo, gmm-demo),
hierarchisches Verschmelzen (agglomerative-demo, hdbscan-demo) - hier zum ersten Mal
**Bayesianische Inferenz per MCMC-Sampling**.

**Bewusste Vereinfachung, offen benannt**: sphärische Komponenten mit bekannter, fester
Varianz σ² (kein voll geschätztes Kovarianz-Konstrukt wie in gmm-demo) und ein
Normal-Normal-konjugierter Prior auf die Mittelwerte. Das hält den eigentlichen neuen
Kerngedanken - automatische Clusterzahl über den Chinese-Restaurant-Process (CRP) - klar
im Fokus, statt ihn mit der (in gmm-demo bereits ausführlich behandelten)
Kovarianzform-Frage zu vermischen. Volle Kovarianzschätzung über eine
Normal-Inverse-Wishart-Konjugation wäre der nächste, hier bewusst nicht gebaute
Ausbauschritt (Analogon zum k-Medoids-Verzicht in kmeans-demo).

## Warum diese Demo anders aufgebaut ist

Bei gmm-demo war die Schwierigkeitsachse die Kovarianz-Annahme. Hier ist es der
**Konzentrationsparameter α** - er ersetzt k vollständig, ist aber selbst kein Free Lunch:

- **Einfaches Beispiel**: klar getrennte Gruppen, moderates α - trifft die wahre
  Clusterzahl zuverlässig, ganz ohne dass sie irgendwo vorgegeben wurde.
- **Zu kleines α (Unterclustering)**: stark überlappende Gruppen, sehr kleines α -
  verschmilzt zu wenigen, großen Clustern.
- **Zu großes α (Überclustering)**: dieselben Gruppen, größeres α - zerfällt in unnötig
  viele kleine Cluster. Live in der "📐"-Sektion nachgewiesen: gefundene vs. wahre
  Clusteranzahl über eine α-Spanne, nicht nur behauptet.
- **Viele Gruppen**: mehr wahre Gruppen gleichzeitig - zeigt, wie robust ein vernünftiges
  α über eine deutlich größere Spanne bleibt, wenn die Gruppen klar getrennt sind.

## Visualisierung

Der Schritt-Regler durchläuft **Sweeps** (ein Sweep = jeder Punkt einmal neu gezogen),
beginnend bei der informationsfreien Startzuordnung "jeder Punkt sein eigenes Cluster".
Die Punktfarbe nutzt ein **dynamisches Farbschema** (HSL-Farbrad ab mehr Clustern als
Palettenfarben, siehe hdbscan-demo, wo ein Modulo-Ansatz hier einen echten Bug verursacht
hätte) - die Clusteranzahl wächst und schrumpft während der Animation selbst, anders als
in jeder anderen Demo dieser Reihe. Das **Clusteranzahl-über-Sweeps**-Diagramm daneben
zeigt live die Stabilisierung zu einer stationären Verteilung - der inhaltliche Gegenpart
zu gmm-demos Log-Likelihood-Kurve.

## Sicherheitsgrenzen

`MAX_SWEEPS` (50) begrenzt die Animationsschritte. Anders als bei k-Means' Fixpunkt oder
EMs Toleranzschwelle gibt es hier **kein** Konvergenzkriterium im klassischen Sinn - MCMC
konvergiert zu einer **stationären Verteilung** über Partitionen, nicht zu einem festen
Zustand, weshalb die Clusteranzahl auch nach vielen Sweeps noch leicht schwanken kann
(bewusst sichtbar gelassen, nicht versteckt).

## Verifikation

- **Handgerechnete Beispiele**: Posterior-Prädiktivverteilung (Normal-Normal-Konjugation)
  und CRP-Prior-Wahrscheinlichkeiten von Hand nachgerechnet.
- **Exakte Verifikation auf einer Spielzeug-Instanz** (die aufwendigste, aber
  aussagekräftigste Prüfung der ganzen Reihe): auf einer winzigen Instanz (n=5) werden
  ALLE möglichen Partitionen enumeriert und ihre exakte Posterior-Wahrscheinlichkeit
  berechnet (Ewens-Sampling-Formel für den CRP-Prior, sequenzielle Prädiktiv-Produkte für
  die marginale Likelihood je Cluster - beide unabhängig von der internen Sampler-Logik
  implementiert). Die empirische Verteilung von 300 unabhängigen Sampler-Läufen wird gegen
  diese exakte Verteilung geprüft (`test_sampler_stationary_distribution_matches_exact_posterior_on_toy_instance`).
- **Struktur-Invarianten**: jede bedingte Verteilung summiert zu 1, jeder Punkt genau
  einem Cluster zugeordnet.
- **Kern-Behauptungen der Demo direkt getestet**: ein vernünftiges α trifft die wahre
  Clusterzahl regelmäßig; zu kleines/großes α führt zu Unter-/Überclustering
  (`test_reasonable_alpha_recovers_true_cluster_count`,
  `test_small_alpha_causes_underclustering`, `test_large_alpha_causes_overclustering`).
- **Alle vier Presets direkt gegen das tatsächliche App-Verhalten getestet** (Szenario-Seed
  UND Sampler-Seed identisch, wie `app.py` es tatsächlich macht) - ein Preset, das nur mit
  einem anderen (Test-eigenen) Sampler-Seed funktioniert, aber live ein anderes Ergebnis
  zeigt, wäre sonst unbemerkt geblieben (siehe project-memory für den konkreten Vorfall).
- `sklearn.mixture.BayesianGaussianMixture` ist in den Tests **nur ein grober,
  qualitativer** Kreuzvergleich (Variational Inference statt Gibbs-Sampling - kein exaktes
  Zahlen-Match zu erwarten), nicht die primäre Korrektheitsprüfung.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Hauptablauf: Presets, Einstellungen, Sweep-Animation, Kleinmultiples, Alpha-Vergleich, Formulierungs-Expander |
| `dp_constants.py` | Defaults, Regler-Grenzen, Sicherheitsgrenzen, `PRESETS` |
| `dp_presets.py` | `SettingSpec`/`SETTING_SPECS`, Permalink-Logik, Presets, Zufalls-Seed-Button |
| `dp_scenario.py` | Zufällige Punktwolken: k kreisförmige, gleich gestreute Gauß-Gruppen auf einem Ring (wie kmeans-demo) |
| `dp_algorithm.py` | Collapsed-Gibbs-Sampler from scratch (CRP-Prior, Normal-Normal-Posterior-Prädiktiv, effiziente inkrementelle Cluster-Buchführung), mit vollständigem Sweep-Protokoll |
| `dp_evaluation.py` | Rand-Index (from scratch), Alpha-Vergleich (gefundene vs. wahre Clusteranzahl) |
| `dp_visualization.py` | Punktwolke mit dynamischem Farbschema, Clusteranzahl-über-Sweeps-Diagramm, Kleinmultiples, Alpha-Vergleichsdiagramm (Plotly) |
| `tests/` | Handinstanzen, exakte Enumerationsverifikation, Struktur-Invarianten, Alpha-Sensitivitäts-Nachweise, Preset-gegen-App-Verhalten-Tests, AppTest-Smoke-Test |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
