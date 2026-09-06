"""Dirichlet-Process-Mixture für Sammel-Routen ohne feste Anzahl UND ohne Kovarianzform-
Frage - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im
Vergleich) zeigt diese Demo EIN Verfahren - ein Dirichlet-Process-Mixture-Modell (DPMM),
berechnet über einen Collapsed-Gibbs-Sampler - und lässt stattdessen die Empfindlichkeit
gegenüber dem Konzentrationsparameter α wachsen. Siebtes Stück der "Konzepte"-Reihe,
Fortsetzung von gmm-demo: behebt GENAU die dort offen benannte verbleibende Schwäche -
die Komponentenzahl k muss nicht mehr vorab feststehen (siehe README für die Einordnung).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import dp_constants as C
from dp_algorithm import run
from dp_evaluation import alpha_comparison
from dp_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from dp_scenario import generate_instance
from dp_visualization import (
    build_alpha_comparison_chart,
    build_cluster_count_chart,
    build_mini_scatter_figure,
    build_scatter_figure,
)

st.set_page_config(page_title="Dirichlet-Process-Mixture – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _compute_run(n_points, k, spread, seed, alpha, sigma):
    instance = generate_instance(n_points, k, spread, seed)
    result = run(instance.as_array(), alpha, sigma ** 2, seed)
    return instance, result


@st.cache_data(show_spinner=False)
def _compute_mini_run(instance, alpha, sigma):
    return run(instance.as_array(), alpha, sigma ** 2, C.COMPARISON_SAMPLER_SEED)


@st.cache_data(show_spinner=False)
def _compute_alpha_comparison(instance, true_k, sigma, current_alpha):
    alpha_values = sorted(set(C.ALPHA_SWEEP_VALUES) | {current_alpha})
    return alpha_comparison(
        instance.as_array(), true_k, alpha_values, sigma ** 2,
        C.COMPARISON_SAMPLER_SEED, C.MAX_SWEEPS,
    )


st.title("🎲 Dirichlet-Process-Mixture: Sammel-Routen ohne feste Anzahl")
st.markdown(
    """
gmm-demo hat gezeigt, wie **Gaussian Mixture Models** k-Means' Annahme kugelförmiger,
gleich gestreuter Cluster und harter Zuweisung beheben - benennt dabei aber offen die
eigene verbleibende Schwäche: die Anzahl der Komponenten k muss weiterhin vorab
feststehen, genau wie bei k-Means. Diese Demo behebt **genau das**, mit einem völlig
anderen Mechanismus als DBSCAN/HDBSCAN (Dichte): einem **Dirichlet-Process-Mixture-Modell**
(DPMM), dem bayesianischen, nichtparametrischen Gegenstück. Statt k geben Sie nur einen
**Konzentrationsparameter α** vor, der beschreibt, wie leicht ein neues Cluster eröffnet
wird - berechnet über einen **Collapsed-Gibbs-Sampler**. Genau **wie** das funktioniert,
erklärt der aufgeklappte Abschnitt direkt darunter - bevor weiter unten live geprüft wird,
ob die gefundene Clusterzahl zur wahren passt.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren "
    "vergleichen, zeigt diese Demo - Teil der wachsenden \"Konzepte\"-Reihe - **ein** "
    "Verfahren an einem wachsenden Beispiel: die Schwierigkeitsachse ist hier der "
    "Konzentrationsparameter α - ein Parameter des einen gezeigten Verfahrens, kein "
    "Methodenvergleich."
)

with st.expander("So funktioniert ein Dirichlet-Process-Mixture", expanded=True):
    st.markdown(
        """
Ein DPMM ersetzt die feste Komponentenzahl k durch einen **Chinese-Restaurant-Process
(CRP)**: stellen Sie sich ein Restaurant mit unendlich vielen Tischen vor. Jeder neue Gast
(Punkt) setzt sich mit einer Wahrscheinlichkeit **proportional zur Tischgröße** zu einem
bereits besetzten Tisch (Cluster) - oder eröffnet mit einer Wahrscheinlichkeit
**proportional zu α** einen neuen, leeren Tisch. Das ist die "rich get richer"-Eigenschaft,
die automatisch eine plausible Clusterzahl erzeugt, ohne dass irgendwo k festgelegt wird.

Berechnet wird das über **Collapsed Gibbs Sampling** (Neal, 2000): jeder Punkt wird
nacheinander gedanklich aus seinem aktuellen Cluster entfernt, dann neu gezogen - gewichtet
nach CRP-Wahrscheinlichkeit **mal** wie gut er zu den übrigen Mitgliedern jedes Clusters
passt (die Cluster-Mittelwerte selbst werden dabei analytisch herausintegriert, daher
"collapsed"). Ein Schritt in der Animation unten ist ein vollständiger **Sweep**
(jeder Punkt einmal neu gezogen).

Wichtiger Unterschied zu jedem bisherigen Verfahren dieser Reihe: MCMC-Sampling
**konvergiert** nicht zu einem festen Endzustand wie k-Means oder EM, sondern zu einer
**stationären Verteilung** über mögliche Clusterzuordnungen - die Clusteranzahl kann sich
deshalb auch nach vielen Sweeps noch leicht ändern, was die Punktwolke und das
Clusteranzahl-Diagramm unten bewusst sichtbar lassen, statt es zu verstecken.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
PRESET_HELP = {
    "Einfaches Beispiel": "Klar getrennte Gruppen, moderates α - trifft die wahre Clusterzahl zuverlässig.",
    "Zu kleines α (Unterclustering)": "Überlappende Gruppen, sehr kleines α - verschmilzt zu wenigen, großen Clustern.",
    "Zu großes α (Überclustering)": "Dieselben Gruppen, größeres α - zerfällt in unnötig viele kleine Cluster.",
    "Viele Gruppen": "Mehr wahre Gruppen gleichzeitig - zeigt wachsende Komplexität.",
}
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_points = st.slider("Anzahl Adressen", *bounds("n_points_slider"), key="n_points_slider")
    k = st.slider("Anzahl wahrer Gruppen", *bounds("k_slider"), key="k_slider")
    spread = st.slider(
        "Streuung", *bounds("spread_slider"), key="spread_slider", step=0.05,
        help="Klein = Gruppen klar getrennt. Groß = Gruppen überlappen sich spürbar.",
    )
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)

    st.markdown("**DPMM-Parameter**")
    alpha = st.slider(
        "α (Konzentration)", *bounds("alpha_slider"), key="alpha_slider", step=0.01,
        help="Ersetzt k komplett: wie leicht ein neuer Tisch/ein neues Cluster eröffnet "
        "wird. Klein = wenige, große Cluster. Groß = viele, kleine Cluster.",
    )
    sigma = st.slider(
        "σ (angenommene Cluster-Streuung)", *bounds("sigma_slider"), key="sigma_slider", step=0.05,
        help="Wie breit ein einzelnes Cluster typischerweise ist - ähnlich einem "
        "'bekannten eps' bei DBSCAN.",
    )

    st.button(
        "🎲 Neue Punktwolke generieren",
        width="stretch",
        on_click=randomize_seed,
        help="Würfelt einen neuen Zufalls-Seed für die Adressen.",
    )

sync_query_params(n_points, k, spread, seed, alpha, sigma)

with st.spinner("Führe Collapsed-Gibbs-Sampling aus..."):
    instance, result = _compute_run(int(n_points), int(k), spread, int(seed), alpha, sigma)

max_step = len(result.sweeps) - 1
run_key = (n_points, k, spread, seed, alpha, sigma)
if "dp_step" not in st.session_state or st.session_state.get("dp_step_owner") != run_key:
    st.session_state["dp_step"] = max_step
    st.session_state["dp_step_owner"] = run_key

st.markdown("## 🎯 Collapsed Gibbs Sampling in Aktion")

step = st.slider(
    "Schritt (Sweep)", 0, max_step, key="dp_step",
    help="Schritt 0 = Startzuordnung (jeder Punkt sein eigenes Cluster), danach je ein "
    "vollständiger Sweep. Anders als bei k-Means/EM gibt es kein festes Konvergenz-Ende -"
    " die letzten Schritte zeigen die stationäre Verteilung, nicht einen Fixpunkt.",
)

current_sweep = result.sweeps[step]
scatter_col, count_col = st.columns(2)
with scatter_col:
    st.plotly_chart(
        build_scatter_figure(instance.as_array(), current_sweep.labels), width="stretch",
        key=f"scatter_{step}",
    )
with count_col:
    st.plotly_chart(
        build_cluster_count_chart(result.cluster_count_trace[: step + 1]), width="stretch",
        key=f"cluster_count_{step}",
    )

lm1, lm2, lm3 = st.columns(3)
lm1.metric("Sweep", current_sweep.sweep)
lm2.metric("Gefundene Clusteranzahl", current_sweep.n_clusters)
lm3.metric("Wahre Gruppenzahl", instance.k)

st.markdown("**Und mit anderen α-Werten?**")
st.caption(
    "Gleiche Adressen wie oben, jeweils mit derselben Sampler-Startkonfiguration - nur α "
    "unterscheidet sich, jeweils als Vielfaches des aktuell eingestellten Werts, am Ende "
    "der Sweep-Kette gezeigt."
)
example_alpha_values = sorted({
    round(min(max(alpha * factor, C.ALPHA_MIN), C.ALPHA_MAX), 4) for factor in (0.1, 0.5, 3.0, 10.0)
})
example_cols = st.columns(len(example_alpha_values))
for col, example_alpha in zip(example_cols, example_alpha_values):
    with col:
        example_result = _compute_mini_run(instance, example_alpha, sigma)
        st.plotly_chart(
            build_mini_scatter_figure(instance.as_array(), example_result.final_labels),
            width="stretch", key=f"mini_{example_alpha}",
        )
        st.caption(f"α={example_alpha:g} · {example_result.final_sweep.n_clusters} Cluster")

st.markdown("---")

st.subheader("📐 Trifft die DP-Mixture die wahre Clusterzahl, ohne dass wir sie vorgeben?")
st.markdown(
    """
Live für Ihr aktuelles Szenario berechnet: die **gefundene Clusteranzahl** über mehrere
α-Werte, gegen die **wahre Gruppenzahl** (rote Linie). α ist selbst ein Parameter - kein
Free Lunch -, aber mit einer interpretierbaren Bedeutung (die "Kosten", ein neues Cluster
zu eröffnen) statt einer reinen Distanzschwelle wie eps bei DBSCAN.
"""
)

comparison = _compute_alpha_comparison(instance, instance.k, sigma, alpha)
st.plotly_chart(build_alpha_comparison_chart(comparison), width="stretch", key="alpha_comparison")

found_at_current = comparison.get(alpha, {}).get("found_k")
if found_at_current == instance.k:
    st.success(
        f"✅ Bei α={alpha:g} trifft die DP-Mixture die wahre Gruppenzahl ({instance.k}) "
        f"exakt - ganz ohne dass k irgendwo vorgegeben wurde."
    )
else:
    st.info(
        "Bei diesem α weicht die gefundene Clusteranzahl von der wahren Gruppenzahl ab - "
        "probieren Sie einen anderen α-Wert (Regler links) oder eines der Presets."
    )

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell**: $x_i \sim \mathcal{N}(\mu_{z_i}, \sigma^2 I)$ mit bekannter Varianz $\sigma^2$,
und einem konjugierten Prior auf die Cluster-Mittelwerte $\mu_c \sim \mathcal{N}(\mu_0,
\tau^2 I)$ (bewusste Vereinfachung gegenüber gmm-demos voll geschätzter Kovarianz - hier
soll die automatische Clusterzahl im Fokus stehen, nicht die Kovarianzform).

**Chinese-Restaurant-Process-Prior**: Punkt $i$ gehört mit Wahrscheinlichkeit proportional
zur Clustergröße $n_{c,-i}$ (ohne Punkt $i$) zu einem bestehenden Cluster, oder mit
Wahrscheinlichkeit proportional zu $\alpha$ zu einem neuen Cluster:

$$
P(z_i = c \mid z_{-i}) \propto n_{c,-i}, \qquad P(z_i = \text{neu} \mid z_{-i}) \propto \alpha
$$

**Posterior-Prädiktivverteilung** (Normal-Normal-Konjugation, Cluster-Mittel
herausintegriert): gegeben $m$ Punkte eines Clusters mit Summe $S$,

$$
\text{post\_var} = \left(\frac{1}{\tau^2} + \frac{m}{\sigma^2}\right)^{-1}, \qquad
\text{post\_mean} = \text{post\_var} \left(\frac{\mu_0}{\tau^2} + \frac{S}{\sigma^2}\right)
$$

$$
x \mid \text{Cluster} \sim \mathcal{N}(\text{post\_mean},\ \text{post\_var} + \sigma^2)
$$

Für ein neues (leeres) Cluster ($m=0$) ist das exakt die Prior-Prädiktivverteilung
$\mathcal{N}(\mu_0, \tau^2+\sigma^2)$.

**Collapsed Gibbs Sampling** (Neal, 2000, "Algorithm 3"): pro Sweep wird jeder Punkt
gedanklich entfernt und neu gezogen, gewichtet mit CRP-Prior mal Posterior-Prädiktiv-
Likelihood über alle bestehenden Cluster plus die Option "neues Cluster".

**Exchangeability**: die gemeinsame Verteilung der Punkte unter dem CRP ist unabhängig von
ihrer Erzeugungsreihenfolge - deshalb lässt sich die marginale Likelihood eines Clusters
als sequenzielles Produkt der Prädiktivverteilung in JEDER beliebigen Reihenfolge berechnen
(genutzt für die exakte Verifikation in `tests/test_algorithm.py`, die alle möglichen
Partitionen einer winzigen Instanz enumeriert und deren exakte Posterior-Wahrscheinlichkeit
gegen die empirische Verteilung vieler Sampler-Läufe prüft).

**Terminierung**: anders als k-Means' Fixpunkt oder EMs Toleranzschwelle konvergiert MCMC
zu einer stationären Verteilung über Partitionen, nicht zu einem festen Zustand - der
Sampler läuft deshalb immer genau `MAX_SWEEPS` Schritte, ohne Toleranzprüfung.

**Möglicher nächster Ausbauschritt** (hier nicht gebaut): volle Kovarianzschätzung über
eine Normal-Inverse-Wishart-Konjugation statt bekannter, sphärischer Varianz - würde
gmm-demos Kovarianzform-Flexibilität mit dieser Demos automatischer Clusterzahl
kombinieren.

Implementiert in `dp_algorithm.py` (Collapsed-Gibbs-Sampler, CRP) und `dp_evaluation.py`
(Rand-Index, Alpha-Vergleich).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
