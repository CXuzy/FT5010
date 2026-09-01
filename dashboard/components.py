from dash import html


def make_metric_card(title, value, accent="#3b82f6", subtitle=None):
    return html.Div(
        [
            html.Div(
                [
                    html.Span(
                        title,
                        style={
                            "fontSize": "13px",
                            "color": "#94a3b8",
                            "fontWeight": "600",
                            "letterSpacing": "0.3px",
                            "textTransform": "uppercase",
                        },
                    ),
                    html.Span(
                        "●",
                        style={
                            "color": accent,
                            "fontSize": "12px",
                            "marginLeft": "8px",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "space-between",
                    "marginBottom": "10px",
                },
            ),
            html.Div(
                value,
                style={
                    "fontSize": "30px",
                    "fontWeight": "700",
                    "color": "#f8fafc",
                    "lineHeight": "1.1",
                    "wordBreak": "break-word",
                },
            ),
            html.Div(
                subtitle or "",
                style={
                    "fontSize": "12px",
                    "color": "#64748b",
                    "marginTop": "8px",
                    "minHeight": "16px",
                },
            ),
        ],
        style={
            "border": "1px solid #1f2a44",
            "borderRadius": "18px",
            "padding": "18px 20px",
            "background": "linear-gradient(180deg, #121a2b 0%, #0f172a 100%)",
            "boxShadow": "0 8px 24px rgba(0,0,0,0.22)",
            "minHeight": "128px",
        },
    )


def make_section_title(title, subtitle=None):
    return html.Div(
        [
            html.H2(
                title,
                style={
                    "margin": "0",
                    "fontSize": "24px",
                    "fontWeight": "700",
                    "color": "#f8fafc",
                },
            ),
            html.Div(
                subtitle or "",
                style={
                    "marginTop": "6px",
                    "fontSize": "13px",
                    "color": "#94a3b8",
                },
            ),
        ],
        style={"marginBottom": "14px"},
    )


def make_status_badge(text, tone="neutral"):
    tone_map = {
        "success": {"bg": "rgba(34,197,94,0.16)", "fg": "#4ade80", "border": "#14532d"},
        "danger": {"bg": "rgba(239,68,68,0.16)", "fg": "#f87171", "border": "#7f1d1d"},
        "warning": {"bg": "rgba(245,158,11,0.16)", "fg": "#fbbf24", "border": "#78350f"},
        "info": {"bg": "rgba(59,130,246,0.16)", "fg": "#60a5fa", "border": "#1e3a8a"},
        "neutral": {"bg": "rgba(148,163,184,0.14)", "fg": "#cbd5e1", "border": "#334155"},
    }
    c = tone_map.get(tone, tone_map["neutral"])

    return html.Span(
        text,
        style={
            "display": "inline-block",
            "padding": "6px 12px",
            "borderRadius": "999px",
            "backgroundColor": c["bg"],
            "color": c["fg"],
            "border": f"1px solid {c['border']}",
            "fontSize": "12px",
            "fontWeight": "700",
            "letterSpacing": "0.2px",
        },
    )


def make_info_row(label, value):
    return html.Div(
        [
            html.Div(
                label,
                style={
                    "color": "#94a3b8",
                    "fontSize": "13px",
                    "fontWeight": "600",
                },
            ),
            html.Div(
                value,
                style={
                    "color": "#f8fafc",
                    "fontSize": "14px",
                    "fontWeight": "600",
                    "textAlign": "right",
                    "wordBreak": "break-word",
                },
            ),
        ],
        style={
            "display": "flex",
            "justifyContent": "space-between",
            "gap": "16px",
            "padding": "10px 0",
            "borderBottom": "1px solid #1e293b",
        },
    )