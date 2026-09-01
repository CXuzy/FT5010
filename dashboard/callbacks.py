import traceback
from datetime import datetime

import plotly.graph_objects as go
from dash import html, Input, Output, State, dash_table

from dashboard.components import (
    make_metric_card,
    make_status_badge,
    make_info_row,
)
from dashboard.services import (
    fmt_num,
    get_account_snapshot,
    get_latest_status,
    get_status_history,
    get_trade_history,
    get_live_positions_df,
    get_live_strategy_state,
)
from dashboard.utils import (
    build_positions_compare_table,
    build_nav_figure,
    build_benchmark_figure,
)
from live_trading.kill_switch import close_all_positions
from live_trading.logger import write_kill_log
from live_trading.config import OANDA_ENV, GRANULARITY, INSTRUMENTS


def _tone_for_pnl(x):
    try:
        return "success" if float(x) >= 0 else "danger"
    except Exception:
        return "neutral"


def _accent_for_pnl(x):
    try:
        return "#22c55e" if float(x) >= 0 else "#ef4444"
    except Exception:
        return "#3b82f6"


def _format_time_short(x):
    if x in [None, "-", ""]:
        return "-"
    try:
        s = str(x)
        if "T" in s:
            return s.replace("T", " ")[:19] + " UTC"
        return s[:19]
    except Exception:
        return str(x)


def _build_strategy_state_block(state):
    if state is None:
        return html.Div(
            "Unable to generate live signal right now.",
            style={
                "padding": "14px",
                "borderRadius": "14px",
                "backgroundColor": "#111827",
                "color": "#f8fafc",
                "border": "1px solid #1e293b",
            },
        )

    signal = state.get("latest_signal", {})
    position = state.get("latest_position", {})

    rows = []
    instruments = sorted(set(list(signal.keys()) + list(position.keys())))
    for inst in instruments:
        rows.append(
            {
                "instrument": inst,
                "signal": signal.get(inst, 0),
                "target_weight": round(float(position.get(inst, 0.0)), 4),
            }
        )

    regime_badge = make_status_badge(
        "BULL" if state.get("is_bull") else "BEAR",
        "success" if state.get("is_bull") else "warning",
    )

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            make_info_row("Timestamp", _format_time_short(state.get("timestamp"))),
                            html.Div(
                                [
                                    html.Div(
                                        "Regime",
                                        style={
                                            "color": "#94a3b8",
                                            "fontSize": "13px",
                                            "fontWeight": "600",
                                        },
                                    ),
                                    regime_badge,
                                ],
                                style={
                                    "display": "flex",
                                    "justifyContent": "space-between",
                                    "alignItems": "center",
                                    "gap": "16px",
                                    "padding": "10px 0",
                                    "borderBottom": "1px solid #1e293b",
                                },
                            ),
                            make_info_row("Basket Index", str(round(float(state.get("basket_index", 0.0)), 6))),
                            make_info_row("Basket MA", str(round(float(state.get("basket_ma", 0.0)), 6))),
                            make_info_row("Realized Vol", str(round(float(state.get("latest_realized_vol", 0.0)), 6))),
                            make_info_row("Leverage", str(round(float(state.get("latest_leverage", 1.0)), 4))),
                        ],
                        style={
                            "border": "1px solid #1e293b",
                            "borderRadius": "16px",
                            "padding": "14px 16px",
                            "backgroundColor": "#111827",
                            "marginBottom": "16px",
                        },
                    ),
                    html.Div(
                        [
                            html.Div(
                                "Signal Table",
                                style={
                                    "fontSize": "14px",
                                    "fontWeight": "700",
                                    "color": "#e2e8f0",
                                    "marginBottom": "10px",
                                },
                            ),
                            dash_table.DataTable(
                                columns=[
                                    {"name": "Instrument", "id": "instrument"},
                                    {"name": "Signal", "id": "signal"},
                                    {"name": "Target Weight", "id": "target_weight"},
                                ],
                                data=rows,
                                style_table={"overflowX": "auto"},
                                style_cell={
                                    "textAlign": "left",
                                    "padding": "10px",
                                    "backgroundColor": "#111827",
                                    "color": "#e2e8f0",
                                    "border": "1px solid #1e293b",
                                    "fontFamily": "Inter, Arial, sans-serif",
                                    "fontSize": "13px",
                                },
                                style_header={
                                    "backgroundColor": "#0f172a",
                                    "color": "#93c5fd",
                                    "fontWeight": "700",
                                    "border": "1px solid #1e293b",
                                },
                                style_data_conditional=[
                                    {
                                        "if": {"filter_query": "{signal} > 0", "column_id": "signal"},
                                        "color": "#4ade80",
                                        "fontWeight": "700",
                                    },
                                    {
                                        "if": {"filter_query": "{signal} < 0", "column_id": "signal"},
                                        "color": "#f87171",
                                        "fontWeight": "700",
                                    },
                                ],
                                page_size=10,
                            ),
                        ]
                    ),
                ]
            )
        ]
    )


def _apply_dark_plot_style(fig, title=None, yaxis_title=None):
    fig.update_layout(
        template=None,
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font={"color": "#e2e8f0"},
        title=title,
        xaxis_title=None,
        yaxis_title=yaxis_title,
        height=420,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(
            bgcolor="rgba(15,23,42,0.0)",
            borderwidth=0,
            font=dict(color="#cbd5e1"),
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#1e293b",
        zeroline=False,
        linecolor="#334155",
        tickfont=dict(color="#94a3b8"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#1e293b",
        zeroline=False,
        linecolor="#334155",
        tickfont=dict(color="#94a3b8"),
    )
    return fig


def register_callbacks(app, client):
    @app.callback(
        [
            Output("top-system-status", "children"),
            Output("last-refresh", "children"),
            Output("card-nav", "children"),
            Output("card-balance", "children"),
            Output("card-unrealized", "children"),
            Output("card-realized", "children"),
            Output("card-margin-used", "children"),
            Output("card-margin-available", "children"),
            Output("card-regime", "children"),
            Output("card-last-status-time", "children"),
            Output("latest-strategy-state", "children"),
            Output("positions-table", "columns"),
            Output("positions-table", "data"),
            Output("nav-graph", "figure"),
            Output("benchmark-graph", "figure"),
            Output("trade-table", "columns"),
            Output("trade-table", "data"),
            Output("footer-system-info", "children"),
            Output("hidden-error", "children"),
        ],
        Input("refresh-interval", "n_intervals")
    )
    def refresh_dashboard(n):
        try:
            account = get_account_snapshot(client)
            latest_status = get_latest_status()
            status_df = get_status_history()
            trade_df = get_trade_history()
            live_positions_df = get_live_positions_df(client)
            positions_rows = build_positions_compare_table(latest_status, live_positions_df)
            state = get_live_strategy_state(client)

            regime_value = latest_status.get("regime_bull")
            regime_text = "BULL" if str(regime_value) == "True" else "BEAR" if str(regime_value) == "False" else "-"
            regime_accent = "#22c55e" if regime_text == "BULL" else "#f59e0b" if regime_text == "BEAR" else "#3b82f6"

            top_status = html.Div(
                [
                    make_status_badge("RUNNING", "success"),
                    html.Span(" ", style={"margin": "0 6px"}),
                    make_status_badge(OANDA_ENV.upper(), "info" if OANDA_ENV == "practice" else "warning"),
                    html.Span(" ", style={"margin": "0 6px"}),
                    make_status_badge(f"BAR {GRANULARITY}", "neutral"),
                ]
            )

            strategy_state_block = _build_strategy_state_block(state)

            pos_columns = [{"name": k.replace("_", " ").title(), "id": k} for k in positions_rows[0].keys()] if positions_rows else []
            pos_data = positions_rows

            if not trade_df.empty:
                show_cols = [c for c in trade_df.columns if c in [
                    "timestamp_utc", "event_type", "instrument", "units", "nav"
                ]]
                if not show_cols:
                    show_cols = list(trade_df.columns)

                trade_show = trade_df[show_cols].copy()
                if "timestamp_utc" in trade_show.columns:
                    trade_show["timestamp_utc"] = trade_show["timestamp_utc"].astype(str).str.slice(0, 19)

                trade_columns = [{"name": c.replace("_", " ").title(), "id": c} for c in trade_show.columns]
                trade_data = trade_show.to_dict("records")
            else:
                trade_columns = [{"name": "Message", "id": "message"}]
                trade_data = [{"message": "No executed trades yet."}]

            nav_fig = build_nav_figure(status_df)
            benchmark_fig = build_benchmark_figure(status_df, client)

            nav_fig = _apply_dark_plot_style(nav_fig, title="Strategy NAV History", yaxis_title="NAV")
            benchmark_fig = _apply_dark_plot_style(benchmark_fig, title="Strategy vs Benchmark", yaxis_title="Indexed Value")

            footer_info = html.Div(
                [
                    html.Div(f"Data Source: OANDA {OANDA_ENV}"),
                    html.Div(f"Granularity: {GRANULARITY}"),
                    html.Div(f"Universe: {', '.join(INSTRUMENTS)}"),
                    html.Div("Session Mode: Fresh live validation run"),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(4, minmax(0, 1fr))",
                    "gap": "12px",
                },
            )

            return (
                top_status,
                f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                make_metric_card("NAV", fmt_num(account["nav"]), accent="#3b82f6", subtitle="Net asset value"),
                make_metric_card("Balance", fmt_num(account["balance"]), accent="#60a5fa", subtitle="Account balance"),
                make_metric_card("Unrealized PnL", fmt_num(account["unrealized"]), accent=_accent_for_pnl(account["unrealized"]), subtitle="Open-position profit and loss"),
                make_metric_card("Realized PnL", fmt_num(account["realized"]), accent=_accent_for_pnl(account["realized"]), subtitle="Closed-position profit and loss"),
                make_metric_card("Margin Used", fmt_num(account["margin_used"]), accent="#f59e0b", subtitle="Currently deployed margin"),
                make_metric_card("Margin Available", fmt_num(account["margin_available"]), accent="#22c55e", subtitle="Remaining available margin"),
                make_metric_card("Regime", regime_text, accent=regime_accent, subtitle="Latest logged market regime"),
                make_metric_card("Latest Status Time", _format_time_short(latest_status.get("timestamp_utc")), accent="#94a3b8", subtitle="Most recent log timestamp"),
                strategy_state_block,
                pos_columns,
                pos_data,
                nav_fig,
                benchmark_fig,
                trade_columns,
                trade_data,
                footer_info,
                "",
            )

        except Exception:
            err = traceback.format_exc()
            empty_fig = go.Figure()
            empty_fig = _apply_dark_plot_style(empty_fig)

            return (
                make_status_badge("ERROR", "danger"),
                f"Dashboard refresh failed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                make_metric_card("NAV", "-", accent="#ef4444"),
                make_metric_card("Balance", "-", accent="#ef4444"),
                make_metric_card("Unrealized PnL", "-", accent="#ef4444"),
                make_metric_card("Realized PnL", "-", accent="#ef4444"),
                make_metric_card("Margin Used", "-", accent="#ef4444"),
                make_metric_card("Margin Available", "-", accent="#ef4444"),
                make_metric_card("Regime", "-", accent="#ef4444"),
                make_metric_card("Latest Status Time", "-", accent="#ef4444"),
                html.Pre(err, style={"whiteSpace": "pre-wrap", "color": "#f87171"}),
                [],
                [],
                empty_fig,
                empty_fig,
                [{"name": "message", "id": "message"}],
                [{"message": "Error loading trade data."}],
                "System footer unavailable due to refresh error.",
                err
            )

    @app.callback(
        Output("kill-output", "children"),
        Input("kill-button", "n_clicks"),
        State("kill-output", "children"),
        prevent_initial_call=True
    )
    def run_kill_switch(n_clicks, current_text):
        try:
            results = close_all_positions(client)

            if not results:
                return "No open positions to close."

            lines = ["Closed positions:"]
            for item in results:
                if isinstance(item, dict):
                    inst = item.get("instrument", "UNKNOWN")
                    resp = item.get("response", {})

                    write_kill_log(
                        instrument=inst,
                        response=resp
                    )

                    lines.append(f"- {inst}: closed")
                else:
                    lines.append(f"- {str(item)}")

            lines.append("")
            lines.append(f"Kill switch executed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            return "\n".join(lines)

        except Exception:
            return "Kill switch failed:\n" + traceback.format_exc()