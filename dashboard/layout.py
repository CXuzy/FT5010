from dash import html, dcc, dash_table

from dashboard.components import make_section_title


def create_layout():
    return html.Div(
        [
            # Header
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                "FT5010 LIVE TRADING DASHBOARD",
                                style={
                                    "fontSize": "13px",
                                    "fontWeight": "700",
                                    "color": "#60a5fa",
                                    "letterSpacing": "1.2px",
                                },
                            ),
                            html.H1(
                                "Cross-Sectional FX Momentum Monitor",
                                style={
                                    "margin": "10px 0 8px 0",
                                    "fontSize": "42px",
                                    "fontWeight": "800",
                                    "color": "#f8fafc",
                                    "lineHeight": "1.1",
                                },
                            ),
                            html.Div(
                                "Institutional-style live monitoring for strategy state, execution, benchmark tracking, and risk control.",
                                style={
                                    "fontSize": "16px",
                                    "color": "#94a3b8",
                                    "maxWidth": "900px",
                                    "lineHeight": "1.6",
                                },
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Div(
                                id="top-system-status",
                                style={"marginBottom": "10px"},
                            ),
                            html.Div(
                                id="last-refresh",
                                style={
                                    "fontSize": "13px",
                                    "color": "#94a3b8",
                                },
                            ),
                        ],
                        style={
                            "padding": "18px 20px",
                            "border": "1px solid #1f2a44",
                            "borderRadius": "18px",
                            "background": "linear-gradient(180deg, #121a2b 0%, #0f172a 100%)",
                            "minWidth": "320px",
                            "boxShadow": "0 8px 24px rgba(0,0,0,0.22)",
                        },
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "2.2fr 1fr",
                    "gap": "20px",
                    "alignItems": "start",
                    "marginBottom": "24px",
                },
            ),

            dcc.Interval(id="refresh-interval", interval=30 * 1000, n_intervals=0),

            # KPI row 1
            html.Div(
                [
                    html.Div(id="card-nav"),
                    html.Div(id="card-balance"),
                    html.Div(id="card-unrealized"),
                    html.Div(id="card-realized"),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(4, minmax(0, 1fr))",
                    "gap": "16px",
                    "marginBottom": "16px",
                },
            ),

            # KPI row 2
            html.Div(
                [
                    html.Div(id="card-margin-used"),
                    html.Div(id="card-margin-available"),
                    html.Div(id="card-regime"),
                    html.Div(id="card-last-status-time"),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(4, minmax(0, 1fr))",
                    "gap": "16px",
                    "marginBottom": "26px",
                },
            ),

            # Charts row
            html.Div(
                [
                    html.Div(
                        [
                            make_section_title(
                                "Strategy NAV History",
                                "Portfolio NAV during the live execution window.",
                            ),
                            dcc.Graph(
                                id="nav-graph",
                                config={"displayModeBar": False},
                                style={"height": "420px"},
                            ),
                        ],
                        style={
                            "border": "1px solid #1f2a44",
                            "borderRadius": "20px",
                            "padding": "18px",
                            "backgroundColor": "#0f172a",
                            "boxShadow": "0 8px 24px rgba(0,0,0,0.22)",
                        },
                    ),
                    html.Div(
                        [
                            make_section_title(
                                "Strategy vs Benchmark",
                                "Indexed strategy performance versus FX basket benchmark.",
                            ),
                            dcc.Graph(
                                id="benchmark-graph",
                                config={"displayModeBar": False},
                                style={"height": "420px"},
                            ),
                        ],
                        style={
                            "border": "1px solid #1f2a44",
                            "borderRadius": "20px",
                            "padding": "18px",
                            "backgroundColor": "#0f172a",
                            "boxShadow": "0 8px 24px rgba(0,0,0,0.22)",
                        },
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "gap": "20px",
                    "marginBottom": "24px",
                },
            ),

            # Strategy + positions row
            html.Div(
                [
                    html.Div(
                        [
                            make_section_title(
                                "Latest Strategy State",
                                "Live signal, regime, risk metrics, and current target exposure.",
                            ),
                            html.Div(id="latest-strategy-state"),
                        ],
                        style={
                            "border": "1px solid #1f2a44",
                            "borderRadius": "20px",
                            "padding": "18px",
                            "backgroundColor": "#0f172a",
                            "boxShadow": "0 8px 24px rgba(0,0,0,0.22)",
                        },
                    ),
                    html.Div(
                        [
                            make_section_title(
                                "Current Positions vs Target",
                                "Compare live holdings against latest target units.",
                            ),
                            dash_table.DataTable(
                                id="positions-table",
                                columns=[],
                                data=[],
                                style_table={"overflowX": "auto"},
                                style_cell={
                                    "textAlign": "left",
                                    "padding": "10px",
                                    "backgroundColor": "#0f172a",
                                    "color": "#e2e8f0",
                                    "border": "1px solid #1e293b",
                                    "fontFamily": "Inter, Arial, sans-serif",
                                    "fontSize": "13px",
                                },
                                style_header={
                                    "backgroundColor": "#111827",
                                    "color": "#93c5fd",
                                    "fontWeight": "700",
                                    "border": "1px solid #1e293b",
                                },
                                style_data_conditional=[
                                    {
                                        "if": {
                                            "filter_query": "{gap_vs_target} > 0",
                                            "column_id": "gap_vs_target",
                                        },
                                        "color": "#fbbf24",
                                        "fontWeight": "700",
                                    },
                                    {
                                        "if": {
                                            "filter_query": "{gap_vs_target} < 0",
                                            "column_id": "gap_vs_target",
                                        },
                                        "color": "#f87171",
                                        "fontWeight": "700",
                                    },
                                    {
                                        "if": {
                                            "filter_query": "{gap_vs_target} = 0",
                                            "column_id": "gap_vs_target",
                                        },
                                        "color": "#4ade80",
                                        "fontWeight": "700",
                                    },
                                ],
                                page_size=10,
                            ),
                        ],
                        style={
                            "border": "1px solid #1f2a44",
                            "borderRadius": "20px",
                            "padding": "18px",
                            "backgroundColor": "#0f172a",
                            "boxShadow": "0 8px 24px rgba(0,0,0,0.22)",
                        },
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1.05fr 1fr",
                    "gap": "20px",
                    "marginBottom": "24px",
                },
            ),

            # Trade log + control row
            html.Div(
                [
                    html.Div(
                        [
                            make_section_title(
                                "Recent Trade Log",
                                "Most recent execution records captured during this live session.",
                            ),
                            dash_table.DataTable(
                                id="trade-table",
                                columns=[],
                                data=[],
                                style_table={"overflowX": "auto"},
                                style_cell={
                                    "textAlign": "left",
                                    "padding": "10px",
                                    "maxWidth": "220px",
                                    "whiteSpace": "normal",
                                    "backgroundColor": "#0f172a",
                                    "color": "#e2e8f0",
                                    "border": "1px solid #1e293b",
                                    "fontFamily": "Inter, Arial, sans-serif",
                                    "fontSize": "13px",
                                },
                                style_header={
                                    "backgroundColor": "#111827",
                                    "color": "#93c5fd",
                                    "fontWeight": "700",
                                    "border": "1px solid #1e293b",
                                },
                                page_size=10,
                            ),
                        ],
                        style={
                            "border": "1px solid #1f2a44",
                            "borderRadius": "20px",
                            "padding": "18px",
                            "backgroundColor": "#0f172a",
                            "boxShadow": "0 8px 24px rgba(0,0,0,0.22)",
                        },
                    ),
                    html.Div(
                        [
                            make_section_title(
                                "Execution Control",
                                "Emergency control panel and system alert output.",
                            ),
                            html.Button(
                                "KILL SWITCH: CLOSE ALL POSITIONS",
                                id="kill-button",
                                n_clicks=0,
                                style={
                                    "width": "100%",
                                    "background": "linear-gradient(135deg, #dc2626 0%, #b91c1c 100%)",
                                    "color": "white",
                                    "border": "none",
                                    "padding": "16px 18px",
                                    "fontSize": "16px",
                                    "borderRadius": "14px",
                                    "cursor": "pointer",
                                    "fontWeight": "800",
                                    "boxShadow": "0 10px 24px rgba(127,29,29,0.35)",
                                    "marginBottom": "16px",
                                },
                            ),
                            html.Div(
                                id="kill-output",
                                style={
                                    "marginTop": "10px",
                                    "whiteSpace": "pre-wrap",
                                    "padding": "16px",
                                    "borderRadius": "14px",
                                    "backgroundColor": "#111827",
                                    "border": "1px solid #1e293b",
                                    "color": "#e2e8f0",
                                    "minHeight": "220px",
                                    "fontSize": "13px",
                                    "lineHeight": "1.6",
                                },
                            ),
                        ],
                        style={
                            "border": "1px solid #1f2a44",
                            "borderRadius": "20px",
                            "padding": "18px",
                            "backgroundColor": "#0f172a",
                            "boxShadow": "0 8px 24px rgba(0,0,0,0.22)",
                        },
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1.5fr 0.9fr",
                    "gap": "20px",
                    "marginBottom": "20px",
                },
            ),

            # Footer
            html.Div(
                id="footer-system-info",
                style={
                    "marginTop": "8px",
                    "padding": "16px 18px",
                    "border": "1px solid #1f2a44",
                    "borderRadius": "16px",
                    "backgroundColor": "#0f172a",
                    "color": "#94a3b8",
                    "fontSize": "13px",
                    "lineHeight": "1.7",
                },
            ),

            html.Div(id="hidden-error", style={"display": "none"}),
        ],
        style={
            "minHeight": "100vh",
            "backgroundColor": "#0b1020",
            "padding": "28px",
            "fontFamily": "Inter, Arial, sans-serif",
        },
    )