"""Shared chart styling so every chart in the app looks consistent (dark theme)."""
import plotly.graph_objects as go

SURFACE = "#1a1a19"
INK = "#ffffff"
INK_2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
AXIS = "#383835"

ACTUAL = "#3987e5"      # series 1 (blue)
PREDICTED = "#d95926"   # series 2 (orange) - always drawn dashed as well

# Single-hue blue ramp, dark -> light (low demand recedes into the dark surface)
SEQUENTIAL = ["#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5",
              "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb"]


def style(fig, height=340, legend=True, y_title=None, x_title=None):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=16, b=8),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK_2, size=13),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=INK_2)),
        hoverlabel=dict(bgcolor="#262624", bordercolor=AXIS, font=dict(color=INK)),
        bargap=0.25,
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickfont=dict(color=MUTED),
                     title=dict(text=x_title, font=dict(color=MUTED)))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=AXIS, tickfont=dict(color=MUTED),
                     title=dict(text=y_title, font=dict(color=MUTED)))
    return fig


def bar(x, y, orientation="v", height=340, hover="%{y:,.0f}", **kw):
    fig = go.Figure(go.Bar(
        x=x, y=y, orientation=orientation, marker=dict(color=ACTUAL, cornerradius=4),
        hovertemplate=hover + "<extra></extra>"))
    return style(fig, height=height, legend=False, **kw)


def actual_vs_predicted(x, actual, predicted, height=340, marker_x=None, **kw):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=actual, name="Actual", mode="lines",
                             line=dict(color=ACTUAL, width=2)))
    fig.add_trace(go.Scatter(x=x, y=predicted, name="Predicted", mode="lines",
                             line=dict(color=PREDICTED, width=2, dash="dash")))
    if marker_x is not None:
        fig.add_vline(x=marker_x, line=dict(color=MUTED, width=1, dash="dot"))
    fig.update_layout(hovermode="x unified")
    fig.update_traces(hovertemplate="%{y:,.1f}")
    return style(fig, height=height, **kw)
