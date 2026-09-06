# Dirichlet-Process-Mixture für Sammel-Routen ohne feste Anzahl – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-dpmm-demo.streamlit.app/)**

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
ein neues Cluster eröffnet wird - berechnet über von Grund auf implementierte
**Variational Inference** (Blei & Jordan, 2006): der Chinese-Restaurant-Process wird als
trunkierter Stick-Breaking-Prozess umformuliert und per Koordinatenaufstieg approximiert.
**Umgebaut 2026-09-06** von einem ursprünglichen Collapsed-Gibbs-Sampler (Neal, 2000) -
korrekt, aber langsam und ohne festen Endzustand - auf diese schnellere, deterministische
Alternative (siehe project-memory für den Anlass). **Ehrlich benannt, anders als beim
Sprung zu [leiden-demo](../leiden-demo)**: es gibt hier keinen so klaren De-facto-Standard
wie Leiden für Netzwerk-Community-Detection - Variational Inference ist die
praxisnähere, schnellere Alternative zu MCMC (und das, was
`sklearn.mixture.BayesianGaussianMixture` tatsächlich implementiert), nicht eine
unumstrittene "beste" Methode.

**Bewusste Vereinfachung, offen benannt**: sphärische Komponenten mit bekannter, fester
Varianz σ² (kein voll geschätztes Kovarianz-Konstrukt wie in gmm-demo) und ein
Normal-Normal-konjugierter Prior auf die Mittelwerte - unverändert gegenüber der
ursprünglichen Fassung. Das hält den eigentlichen neuen Kerngedanken - automatische
Clusterzahl über den Chinese-Restaurant-Process (CRP) - klar im Fokus, statt ihn mit der
(in gmm-demo bereits ausführlich behandelten) Kovarianzform-Frage zu vermischen. Volle
Kovarianzschätzung über eine Normal-Inverse-Wishart-Konjugation wäre der nächste, hier
bewusst nicht gebaute Ausbauschritt (Analogon zum k-Medoids-Verzicht in kmeans-demo).

## Warum diese Demo anders aufgebaut ist

Bei gmm-demo war die Schwierigkeitsachse die Kovarianz-Annahme. Hier ist es der
**Konzentrationsparameter α** - er ersetzt k vollständig, ist aber selbst kein Free Lunch,
plus die **Trunkierungsgrenze T** als neuer, ehrlich benannter Preis für Variational
Inference:

- **Einfaches Beispiel**: klar getrennte Gruppen, moderates α - trifft die wahre
  Clusterzahl zuverlässig, ganz ohne dass sie irgendwo vorgegeben wurde.
- **Zu kleines α (Unterclustering)**: überlappende Gruppen, sehr kleines α - verschmilzt
  zu wenigen, großen Clustern.
- **Zu großes α (Überclustering)**: eine andere, gut getrennte Szenerie, größeres α -
  zerfällt in unnötig viele kleine Cluster. Live in der "📐"-Sektion nachgewiesen:
  gefundene vs. wahre Clusteranzahl über eine α-Spanne, nicht nur behauptet.
- **Trunkierung zu niedrig**: die Trunkierungsgrenze T liegt UNTER der wahren
  Gruppenzahl - deckelt die gefundene Clusteranzahl hart, unabhängig von α. Ein
  Fehlermodus, den der ursprüngliche (asymptotisch unbegrenzte) Collapsed-Gibbs-Sampler
  gar nicht kannte.
- **Viele Gruppen**: mehr wahre Gruppen gleichzeitig - zeigt, wie robust ein vernünftiges
  α über eine deutlich größere Spanne bleibt, wenn die Gruppen klar getrennt sind.

## Visualisierung

Der Schritt-Regler durchläuft **Iterationen** eines Koordinatenaufstiegs, beginnend bei
den Responsibilities auf den Startparametern. Die Punktfarbe nutzt ein **dynamisches
Farbschema** (HSL-Farbrad ab mehr Clustern als Palettenfarben, siehe hdbscan-demo, wo ein
Modulo-Ansatz hier einen echten Bug verursacht hätte) - die aktive Komponentenanzahl kann
während der Animation wachsen und schrumpfen, anders als in jeder anderen Demo dieser
Reihe. Das **Clusteranzahl-über-Iterationen**-Diagramm daneben zeigt live die Konvergenz
zu einem festen Endzustand - anders als beim ursprünglichen MCMC-Sampler ein ECHTES
Konvergenzende, nicht nur eine sich stabilisierende Verteilung.

## Sicherheitsgrenzen

`MAX_ITERATIONS` (100) begrenzt die Animationsschritte als reine Absicherung - Variational
Inference hat (anders als der ursprüngliche MCMC-Sampler) ein echtes Toleranzkriterium
(`ELBO_TOL`) und konvergiert normalerweise deutlich früher zu einem festen Endzustand.

## Verifikation

- **KL-Divergenz-Formeln direkt getestet**: $\text{KL}(q\|q)=0$ für identische
  Verteilungen (Beta wie Normal), ein nicht-trivialer Fall zusätzlich numerisch gegen
  `scipy.integrate.quad`/Monte-Carlo-Schätzung geprüft.
- **ELBO-Monotonie** (Struktur-Invariante, exakt und algorithmusunabhängig vom Zufall):
  Koordinatenaufstieg garantiert, dass die ELBO bei jedem Schritt nicht sinkt - über
  mehrere Zufallsszenarien getestet.
- **Exakte Verifikation auf einer Spielzeug-Instanz, neu interpretiert** (die
  aufwendigste, aber aussagekräftigste Prüfung der ganzen Reihe): auf einer winzigen
  Instanz (n=5) wird die exakte MAP-Partition durch Enumeration ALLER möglichen
  Partitionen berechnet (Ewens-Sampling-Formel für den CRP-Prior, sequenzielle
  Prädiktiv-Produkte für die marginale Likelihood je Cluster - Eigenschaften des Modells,
  nicht der Inferenzmethode). Geprüft wird, ob Variational Inference bei den meisten
  Zufalls-Initialisierungen zu genau dieser exakten MAP-Partition konvergiert - ein
  direkter "wie gut ist die Approximation"-Nachweis, den der frühere samplingbasierte
  Ansatz (Vergleich gegen eine empirische Verteilung vieler MCMC-Läufe) so nicht bieten
  konnte.
- **Struktur-Invarianten**: Responsibilities summieren zu 1 je Punkt, die
  Trunkierungsgrenze deckelt die aktive Komponentenanzahl hart.
- **Kern-Behauptungen der Demo direkt getestet**: ein vernünftiges α trifft die wahre
  Clusterzahl; zu kleines/großes α führt zu Unter-/Überclustering; eine zu niedrige
  Trunkierungsgrenze deckelt die Clusteranzahl unabhängig von α
  (`test_reasonable_alpha_recovers_true_cluster_count`,
  `test_small_alpha_causes_underclustering`, `test_large_alpha_causes_overclustering`,
  `test_low_truncation_hard_caps_found_cluster_count`).
- **Alle fünf Presets direkt gegen das tatsächliche App-Verhalten getestet** (Szenario-Seed
  UND Inferenz-Seed identisch, wie `app.py` es tatsächlich macht) - ein Preset, das nur mit
  einem anderen (Test-eigenen) Seed funktioniert, aber live ein anderes Ergebnis zeigt,
  wäre sonst unbemerkt geblieben (siehe project-memory für den konkreten Vorfall).
- `sklearn.mixture.BayesianGaussianMixture` ist in den Tests weiterhin **nur ein grober,
  qualitativer** Kreuzvergleich (volle statt sphärischer/bekannter Kovarianz - kein
  exaktes Zahlen-Match zu erwarten), methodisch aber treffender als zuvor: beide sind
  jetzt tatsächlich Variational Inference für eine DP-Mixture.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Hauptablauf: Presets, Einstellungen, Iterations-Animation, Kleinmultiples, Alpha-Vergleich, Formulierungs-Expander |
| `dp_constants.py` | Defaults, Regler-Grenzen, Sicherheitsgrenzen, `PRESETS` |
| `dp_presets.py` | `SettingSpec`/`SETTING_SPECS`, Permalink-Logik, Presets, Zufalls-Seed-Button |
| `dp_scenario.py` | Zufällige Punktwolken: k kreisförmige, gleich gestreute Gauß-Gruppen auf einem Ring (wie kmeans-demo) |
| `dp_algorithm.py` | Variational Inference from scratch (trunkierter Stick-Breaking-Prozess, Responsibility-/Beta-/Normal-Updates, ELBO), mit vollständigem Iterations-Protokoll |
| `dp_evaluation.py` | Rand-Index (from scratch), Alpha-Vergleich (gefundene vs. wahre Clusteranzahl) |
| `dp_visualization.py` | Punktwolke mit dynamischem Farbschema, Clusteranzahl-über-Iterationen-Diagramm, Kleinmultiples, Alpha-Vergleichsdiagramm (Plotly) |
| `tests/` | KL-Formel-Handbeispiele, ELBO-Monotonie, exakte MAP-Partitions-Verifikation, Struktur-Invarianten, Alpha-/Trunkierungs-Sensitivitäts-Nachweise, Preset-gegen-App-Verhalten-Tests, AppTest-Smoke-Test |

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
