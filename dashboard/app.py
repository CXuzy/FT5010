from dash import Dash

from dashboard.callbacks import register_callbacks
from dashboard.layout import create_layout
from live_trading.config import OANDA_API_KEY, OANDA_ACCOUNT_ID, OANDA_ENV
from live_trading.oanda_client import OandaClient

client = OandaClient(
    api_key=OANDA_API_KEY,
    account_id=OANDA_ACCOUNT_ID,
    env=OANDA_ENV
)

app = Dash(__name__)
app.title = "FT5010 Live Trading Dashboard"
app.layout = create_layout()

register_callbacks(app, client)

if __name__ == "__main__":
    app.run(debug=True, port=8050)