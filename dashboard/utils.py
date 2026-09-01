import plotly.graph_objects as go

from live_trading.config import INSTRUMENTS
from live_trading.live_signal_engine import generate_live_signal


def build_positions_compare_table(latest_status, live_positions_df):
    target_units = latest_status.get("target_units", {})
    logged_positions = latest_status.get("current_positions", {})

    rows = []
    for inst in INSTRUMENTS:
        live_units = 0
        if not live_positions_df.empty:
            sub = live_positions_df.loc[live_positions_df["instrument"] == inst, "current_units_live"]
            if len(sub) > 0:
                live_units = int(sub.iloc[0])

        tgt = int(target_units.get(inst, 0))
        logged = int(logged_positions.get(inst, 0))

        rows.append({
            "instrument": inst,
            "current_units_logged": logged,
            "current_units_live": live_units,
            "target_units": tgt,
            "gap_vs_target": tgt - live_units
        })

    return rows


def build_nav_figure(status_df):
    fig = go.Figure()

    if status_df.empty or "nav" not in status_df.columns:
        fig.update_layout(
            title="Strategy NAV History",
            template="plotly_white",
            xaxis_title="Time",
            yaxis_title="NAV",
            height=420
        )
        return fig

    plot_df = status_df.dropna(subset=["timestamp_utc", "nav"]).copy()
    if plot_df.empty:
        fig.update_layout(
            title="Strategy NAV History",
            template="plotly_white",
            xaxis_title="Time",
            yaxis_title="NAV",
            height=420
        )
        return fig

    fig.add_trace(go.Scatter(
        x=plot_df["timestamp_utc"],
        y=plot_df["nav"],
        mode="lines+markers",
        name="Strategy NAV"
    ))

    fig.update_layout(
        title="Strategy NAV History",
        template="plotly_white",
        xaxis_title="Time",
        yaxis_title="NAV",
        height=420
    )
    return fig


def build_benchmark_figure(status_df, client):
    fig = go.Figure()

    if status_df.empty or "nav" not in status_df.columns:
        fig.update_layout(
            title="Strategy vs Benchmark",
            template="plotly_white",
            xaxis_title="Time",
            yaxis_title="Indexed Value",
            height=420
        )
        return fig

    plot_df = status_df.dropna(subset=["timestamp_utc", "nav"]).copy()
    if plot_df.empty:
        fig.update_layout(
            title="Strategy vs Benchmark",
            template="plotly_white",
            xaxis_title="Time",
            yaxis_title="Indexed Value",
            height=420
        )
        return fig

    plot_df = plot_df.reset_index(drop=True)
    plot_df["strategy_index"] = plot_df["nav"] / float(plot_df["nav"].iloc[0])

    fig.add_trace(go.Scatter(
        x=plot_df["timestamp_utc"],
        y=plot_df["strategy_index"],
        mode="lines+markers",
        name="Strategy"
    ))

    try:
        prices, _ = generate_live_signal(client)
        fx_cols = [c for c in ["EURUSD", "GBPUSD", "USDJPY"] if c in prices.columns]
        if fx_cols:
            bench = prices[fx_cols].copy().tail(len(plot_df)).reset_index(drop=True)

            for c in fx_cols:
                bench[c] = bench[c] / bench[c].iloc[0]

            bench["basket"] = bench.mean(axis=1)

            aligned_len = min(len(plot_df), len(bench))
            fig.add_trace(go.Scatter(
                x=plot_df["timestamp_utc"].iloc[-aligned_len:],
                y=bench["basket"].iloc[-aligned_len:],
                mode="lines",
                name="FX Basket Benchmark"
            ))
    except Exception:
        pass

    fig.update_layout(
        title="Strategy vs Benchmark",
        template="plotly_white",
        xaxis_title="Time",
        yaxis_title="Indexed Value",
        height=420
    )
    return fig