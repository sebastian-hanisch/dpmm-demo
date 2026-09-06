"""Dirichlet-Process-Mixture für Sammel-Routen ohne feste Anzahl UND ohne Kovarianzform-
Frage - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im
Vergleich) zeigt diese Demo EIN Verfahren - ein Dirichlet-Process-Mixture-Modell (DPMM),
berechnet über Variational Inference (Blei & Jordan, 2006) - und lässt stattdessen die
Empfindlichkeit gegenüber dem Konzentrationsparameter α wachsen. Siebtes Stück der
"Konzepte"-Reihe, Fortsetzung von gmm-demo: behebt GENAU die dort offen benannte
verbleibende Schwäche - die Komponentenzahl k muss nicht mehr vorab feststehen (siehe
README für die Einordnung).

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
def _compute_run(n_points, k, spread, seed, alpha, sigma, truncation):
    instance = generate_instance(n_points, k, spread, seed)
    result = run(instance.as_array(), alpha, sigma ** 2, truncation, seed)
    return instance, result


@st.cache_data(show_spinner=False)
def _compute_mini_run(instance, alpha, sigma, truncation):
    return run(instance.as_array(), alpha, sigma ** 2, truncation, C.COMPARISON_SAMPLER_SEED)


@st.cache_data(show_spinner=False)
def _compute_alpha_comparison(instance, true_k, sigma, truncation, current_alpha):
    alpha_values = sorted(set(C.ALPHA_SWEEP_VALUES) | {current_alpha})
    return alpha_comparison(
        instance.as_array(), true_k, alpha_values, sigma ** 2, truncation,
        C.COMPARISON_SAMPLER_SEED, C.MAX_ITERATIONS,
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
wird - berechnet über **Variational Inference** (Blei & Jordan, 2006). Genau **wie** das
funktioniert, erklärt der aufgeklappte Abschnitt direkt darunter - bevor weiter unten live
geprüft wird, ob die gefundene Clusterzahl zur wahren passt.
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

Berechnet wird das hier über **Variational Inference** (Blei & Jordan, 2006): der CRP wird
als **Stick-Breaking-Prozess** umformuliert (unendlich viele Mischgewichte, die aus
"abgebrochenen Stücken" eines Stocks der Länge 1 entstehen) und bei einer **Trunkierungsgrenze
T** abgeschnitten - eine deterministische Optimierung (Koordinatenaufstieg, wie EM in
gmm-demo) nähert die wahre Posterior-Verteilung an, statt sie über MCMC-Sampling
asymptotisch exakt, aber langsam zu erreichen. Diese Demo hat ursprünglich einen
**Collapsed-Gibbs-Sampler** (Neal, 2000) verwendet - exakt, aber deutlich langsamer und
ohne festen Endzustand (siehe Mathe-Abschnitt für die Gegenüberstellung beider Ansätze).

**Ehrlicher Hinweis, anders als beim Sprung zu Leiden in der Spectral-Linie**: es gibt in
der Bayesianischen nichtparametrischen Clustering-Welt keinen so klaren De-facto-Standard
wie Leiden für Netzwerk-Community-Detection. Variational Inference ist die schnellere,
praxisnähere Alternative zu MCMC (und das, was `sklearn.mixture.BayesianGaussianMixture`
tatsächlich implementiert) - aber ebenso oft greifen Praktiker zu einfacheren Heuristiken
(BIC/AIC über ein Standard-GMM). Der Wechsel hier ist also eine Geschwindigkeits- und
Praxis-Entscheidung, keine Behauptung, dies sei "die" unumstrittene beste Methode.

Ein Schritt in der Animation unten ist ein vollständiger **Koordinatenaufstiegs-Zyklus**.
Anders als MCMC-Sampling (stationäre Verteilung, kein Fixpunkt) **konvergiert** Variational
Inference zu einem echten Fixpunkt - wie EM in gmm-demo, mit einem Toleranzkriterium auf
die **ELBO** (die zu optimierende untere Schranke der Modell-Evidenz).
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
PRESET_HELP = {
    "Einfaches Beispiel": "Klar getrennte Gruppen, moderates α - trifft die wahre Clusterzahl zuverlässig.",
    "Zu kleines α (Unterclustering)": "Überlappende Gruppen, sehr kleines α - verschmilzt zu wenigen, großen Clustern.",
    "Zu großes α (Überclustering)": "Dieselben Gruppen, größeres α - zerfällt in unnötig viele kleine Cluster.",
    "Trunkierung zu niedrig": "Die Trunkierungsgrenze T ist kleiner als die wahre Gruppenzahl - deckelt die gefundene Clusteranzahl hart, unabhängig von α.",
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
    truncation = st.slider(
        "T (Trunkierungsgrenze)", *bounds("truncation_slider"), key="truncation_slider",
        help="Obergrenze für die Anzahl möglicher Cluster - der neue Preis für "
        "Variational Inference statt MCMC. Zu niedrig gewählt deckelt die gefundene "
        "Clusteranzahl hart, unabhängig von α.",
    )

    st.button(
        "🎲 Neue Punktwolke generieren",
        width="stretch",
        on_click=randomize_seed,
        help="Würfelt einen neuen Zufalls-Seed für die Adressen.",
    )

sync_query_params(n_points, k, spread, seed, alpha, sigma, truncation)

with st.spinner("Führe Variational Inference aus..."):
    instance, result = _compute_run(int(n_points), int(k), spread, int(seed), alpha, sigma, int(truncation))

max_step = len(result.steps) - 1
run_key = (n_points, k, spread, seed, alpha, sigma, truncation)
if "dp_step" not in st.session_state or st.session_state.get("dp_step_owner") != run_key:
    st.session_state["dp_step"] = max_step
    st.session_state["dp_step_owner"] = run_key

st.markdown("## 🎯 Variational Inference in Aktion")

step = st.slider(
    "Schritt (Iteration)", 0, max_step, key="dp_step",
    help="Schritt 0 = Responsibilities auf den Startparametern, danach je ein "
    "vollständiger Koordinatenaufstiegs-Zyklus. Reglerposition steht standardmäßig auf "
    "dem letzten (konvergierten) Schritt - anders als beim vorherigen MCMC-Sampler gibt "
    "es hier ein ECHTES Konvergenzende.",
)

current_step = result.steps[step]
scatter_col, count_col = st.columns(2)
with scatter_col:
    st.plotly_chart(
        build_scatter_figure(instance.as_array(), current_step.hard_labels), width="stretch",
        key=f"scatter_{step}",
    )
with count_col:
    st.plotly_chart(
        build_cluster_count_chart(result.cluster_count_trace[: step + 1]), width="stretch",
        key=f"cluster_count_{step}",
    )

lm1, lm2, lm3 = st.columns(3)
lm1.metric("Iteration", current_step.iteration)
lm2.metric("Gefundene Clusteranzahl", current_step.n_active_components)
lm3.metric("Wahre Gruppenzahl", instance.k)

st.markdown("**Und mit anderen α-Werten?**")
st.caption(
    "Gleiche Adressen wie oben, jeweils mit derselben Startkonfiguration - nur α "
    "unterscheidet sich, jeweils bei Konvergenz gezeigt."
)
example_alpha_values = sorted({
    round(min(max(alpha * factor, C.ALPHA_MIN), C.ALPHA_MAX), 4) for factor in (0.1, 0.5, 3.0, 10.0)
})
example_cols = st.columns(len(example_alpha_values))
for col, example_alpha in zip(example_cols, example_alpha_values):
    with col:
        example_result = _compute_mini_run(instance, example_alpha, sigma, int(truncation))
        st.plotly_chart(
            build_mini_scatter_figure(instance.as_array(), example_result.final_labels),
            width="stretch", key=f"mini_{example_alpha}",
        )
        st.caption(f"α={example_alpha:g} · {example_result.final_step.n_active_components} Cluster")

st.markdown("---")

st.subheader("📐 Trifft die DP-Mixture die wahre Clusterzahl, ohne dass wir sie vorgeben?")
st.markdown(
    """
Live für Ihr aktuelles Szenario berechnet: die **gefundene Clusteranzahl** über mehrere
α-Werte, gegen die **wahre Gruppenzahl** (rote Linie). α ist selbst ein Parameter - kein
Free Lunch -, aber mit einer interpretierbaren Bedeutung (die "Kosten", ein neues Cluster
zu eröffnen) statt einer reinen Distanzschwelle wie eps bei DBSCAN. Die Trunkierungsgrenze
T (Seitenleiste) begrenzt dabei zusätzlich hart, wie viele Cluster überhaupt möglich sind.
"""
)

comparison = _compute_alpha_comparison(instance, instance.k, sigma, int(truncation), alpha)
st.plotly_chart(build_alpha_comparison_chart(comparison), width="stretch", key="alpha_comparison")

found_at_current = comparison.get(alpha, {}).get("found_k")
if found_at_current == instance.k:
    st.success(
        f"✅ Bei α={alpha:g} trifft die DP-Mixture die wahre Gruppenzahl ({instance.k}) "
        f"exakt - ganz ohne dass k irgendwo vorgegeben wurde."
    )
elif int(truncation) < instance.k:
    st.warning(
        f"⚠️ Die Trunkierungsgrenze T={int(truncation)} liegt UNTER der wahren "
        f"Gruppenzahl ({instance.k}) - mehr als {int(truncation)} Cluster kann das "
        f"Modell hier gar nicht finden, unabhängig von α. Das ist der neue Preis für "
        f"Variational Inference: erhöhen Sie T in der Seitenleiste."
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
soll die automatische Clusterzahl im Fokus stehen, nicht die Kovarianzform). Dieses Modell
ist unverändert gegenüber der ursprünglichen Fassung dieser Demo - nur die Inferenzmethode
hat sich geändert.

**Stick-Breaking statt Chinese-Restaurant-Process**: der CRP-Prior lässt sich äquivalent
als **trunkierter Stick-Breaking-Prozess** schreiben - $\beta_t \sim \text{Beta}(1,\alpha)$
für $t=1,\dots,T-1$, $\beta_T=1$ (Trunkierung bei $T$), Mischgewichte
$\pi_t = \beta_t \prod_{s<t}(1-\beta_s)$.

**Variationsfamilie** (mean-field): $q(\beta_t)=\text{Beta}(\gamma_{t,1},\gamma_{t,2})$,
$q(\mu_t)=\mathcal{N}(m_t,s_t^2 I)$, $q(z_i)=\text{Categorical}(\phi_i)$.

**Koordinatenaufstieg** (ein Schritt = ein voller Zyklus):

$$
\phi_{i,t} \propto \exp\!\big(\mathbb{E}[\log\pi_t] + \mathbb{E}[\log\mathcal{N}(x_i\mid\mu_t,\sigma^2)]\big),
\qquad
\gamma_{t,1}=1+N_t,\ \ \gamma_{t,2}=\alpha+\!\!\sum_{s>t}N_s
$$

$$
s_t^2 = \left(\frac{1}{\tau^2}+\frac{N_t}{\sigma^2}\right)^{-1}, \qquad
m_t = s_t^2\left(\frac{\mu_0}{\tau^2}+\frac{\sum_i\phi_{i,t}x_i}{\sigma^2}\right)
$$

mit $N_t=\sum_i\phi_{i,t}$ - erkennbar dieselbe Normal-Normal-Update-Struktur wie zuvor,
nur mit responsibility-gewichteten statt harten Cluster-Summen.

**ELBO** (die zu maximierende untere Schranke): erwartete Log-Likelihood plus erwarteter
Stick-Breaking-Log-Prior minus Responsibility-Entropie, minus
$\text{KL}(q(\beta_t)\|\text{Beta}(1,\alpha))$ minus $\text{KL}(q(\mu_t)\|\mathcal{N}(\mu_0,\tau^2))$
(beide in geschlossener Form). Koordinatenaufstieg garantiert, dass die ELBO bei jedem
Schritt nicht sinkt (siehe `tests/test_algorithm.py`).

**Vorher: Collapsed Gibbs Sampling** (Neal, 2000, "Algorithm 3") - die ursprüngliche
Fassung dieser Demo: pro Sweep wird jeder Punkt gedanklich entfernt und neu gezogen,
gewichtet mit CRP-Prior mal Posterior-Prädiktiv-Likelihood. **Asymptotisch exakt**
(konvergiert zur wahren Posterior-VERTEILUNG über Partitionen), aber langsam und ohne
festen Endzustand. Variational Inference **approximiert** dieselbe Posterior durch die
oben genannte faktorisierte Familie - schneller, deterministisch (bei fester
Initialisierung), mit einem echten Fixpunkt, aber eben nur eine Näherung.

**Exakte Verifikation, unabhängig von der Inferenzmethode**: die gemeinsame Verteilung der
Punkte unter dem CRP ist unabhängig von ihrer Erzeugungsreihenfolge (Exchangeability) -
deshalb lässt sich für eine winzige Instanz die exakte MAP-Partition durch Enumeration
ALLER möglichen Partitionen berechnen. `tests/test_algorithm.py` prüft, ob Variational
Inference bei den meisten Zufalls-Initialisierungen zu genau dieser exakten MAP-Partition
konvergiert - ein direkter "wie gut ist die Approximation"-Nachweis, den der
samplingbasierte Vorgänger so nicht bieten konnte.

**Ehrlicher Hinweis**: anders als der Sprung zu Leiden in der Spectral-Linie ist dieser
Wechsel KEIN Sprung zu einem unumstrittenen De-facto-Standard - er ist eine
Geschwindigkeits-/Praxis-Entscheidung (siehe Intro-Abschnitt oben) mit einem neuen Preis,
der Trunkierungsgrenze $T$.

Implementiert in `dp_algorithm.py` (Variational Inference, Stick-Breaking, ELBO) und
`dp_evaluation.py` (Rand-Index, Alpha-Vergleich).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
