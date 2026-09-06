"""Plotly-Visualisierungen: Punktwolke mit DYNAMISCHEM Farbschema (die Clusteranzahl
kann von Sweep zu Sweep wachsen/schrumpfen - anders als in jeder anderen Demo dieser
Reihe, wo die Clusterzahl je Ansicht fest ist), das Clusteranzahl-über-Sweeps-Diagramm
(zeigt die Stabilisierung des Samplers), Kleinmultiples und der Alpha-Vergleich."""

import numpy as np

CLUSTER_PALETTE = [
    "#1f77b4", "#d68a2e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]


def _cluster_color(index, n_clusters):
    """Bei bis zu len(CLUSTER_PALETTE) Clustern die feste, qualitative Palette (wie in
    den übrigen Demos). Da die Clusteranzahl hier während der Animation selbst wächst
    und schrumpft (siehe Moduldocstring), reicht ein einfaches Modulo auf die Palette
    nicht - ab mehr Clustern als Palettenfarben ein Farbrad (HSL, gleichmäßig verteilt),
    das für jede Clusteranzahl paarweise unterschiedliche Farben garantiert (siehe
    hdbscan-demo, wo genau dieses Modulo-Problem einen echten Bug verursachte)."""
    if n_clusters <= len(CLUSTER_PALETTE):
        return CLUSTER_PALETTE[index % len(CLUSTER_PALETTE)]
    hue = (index * 360.0 / n_clusters) % 360
    return f"hsl({hue:.1f}, 65%, 50%)"


def _axis_range(data):
    xmin, xmax = data[:, 0].min(), data[:, 0].max()
    ymin, ymax = data[:, 1].min(), data[:, 1].max()
    padx = (xmax - xmin) * 0.1 or 1.0
    pady = (ymax - ymin) * 0.1 or 1.0
    return [xmin - padx, xmax + padx], [ymin - pady, ymax + pady]


def _cluster_traces(data, labels, legend):
    import plotly.graph_objects as go

    traces = []
    cluster_ids = sorted(set(labels.tolist()))
    n_clusters = len(cluster_ids)
    show_legend = legend and n_clusters <= 12
    for index, cid in enumerate(cluster_ids):
        mask = labels == cid
        color = _cluster_color(index, n_clusters)
        traces.append(
            go.Scatter(
                x=data[mask, 0], y=data[mask, 1], mode="markers", name=f"Cluster {cid + 1}",
                showlegend=show_legend,
                marker=dict(color=color, size=7, line=dict(width=0.5, color="white")),
                hoverinfo="skip",
            )
        )
    return traces


def _build_figure(data, labels, height, legend):
    import plotly.graph_objects as go

    fig = go.Figure()
    for trace in _cluster_traces(data, np.asarray(labels), legend):
        fig.add_trace(trace)

    xr, yr = _axis_range(data)
    layout_kwargs = dict(
        template="plotly_white", height=height,
        xaxis=dict(visible=False, range=xr, fixedrange=True),
        yaxis=dict(visible=False, range=yr, fixedrange=True, scaleanchor="x", scaleratio=1),
        showlegend=legend,
        margin=dict(t=40 if legend else 5, l=10 if legend else 5, r=10 if legend else 5, b=10 if legend else 5),
    )
    if legend:
        layout_kwargs["legend"] = dict(orientation="h", yanchor="bottom", y=1.02, x=0)
    fig.update_layout(**layout_kwargs)
    return fig


def build_scatter_figure(data, labels):
    return _build_figure(data, labels, height=460, legend=True)


def build_mini_scatter_figure(data, labels):
    return _build_figure(data, labels, height=220, legend=False)


def build_cluster_count_chart(cluster_count_trace):
    import plotly.graph_objects as go

    sweeps = list(range(len(cluster_count_trace)))
    fig = go.Figure(
        go.Scatter(x=sweeps, y=list(cluster_count_trace), mode="lines+markers", line=dict(color="#1f77b4"))
    )
    fig.update_layout(
        template="plotly_white", height=220,
        xaxis=dict(title="Sweep", fixedrange=True),
        yaxis=dict(title="Gefundene Clusteranzahl", fixedrange=True, dtick=1),
        margin=dict(t=20, l=10, r=10, b=10), showlegend=False,
    )
    return fig


def build_alpha_comparison_chart(results):
    import plotly.graph_objects as go

    alphas = list(results.keys())
    found = [results[a]["found_k"] for a in alphas]
    true_k = results[alphas[0]]["true_k"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=[f"{a:g}" for a in alphas], y=found, name="Gefundene Clusteranzahl", marker_color="#1f77b4")
    )
    fig.add_hline(y=true_k, line=dict(color="#d62728", dash="dash"), annotation_text="Wahre Gruppenzahl")
    fig.update_layout(
        template="plotly_white", height=280,
        xaxis=dict(title="α (Konzentrationsparameter)", fixedrange=True),
        yaxis=dict(title="Clusteranzahl", fixedrange=True, dtick=1),
        margin=dict(t=20, l=10, r=10, b=10), showlegend=False,
    )
    return fig
