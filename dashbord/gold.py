import dash
from dash import dcc, html, callback, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
import pandas as pd
from pycaret.regression import *
import os
import joblib  

app = dash.Dash(__name__, title="Gold Price Prediction Dashboard")
server = app.server

FILE_PATH = r"D:/Term Pr/website_pm25/gold_pre.csv"
MODEL_PATH = r"D:/Term Pr/website_pm25/lightgbm_model_gold_1.pkl"

if MODEL_PATH.endswith(".pkl"):
    MODEL_PATH_PYCARET = MODEL_PATH[:-4]
else:
    MODEL_PATH_PYCARET = MODEL_PATH

app.layout = html.Div(
    [
        html.H1(
            "Gold Price Prediction Dashboard",
            style={"textAlign": "center", "color": "#856404", "marginTop": 20},
        ),
        html.Div(
            [
                html.Label("Number of days to predict:"),
                dcc.Input(
                    id="days-input", type="number", value=9, min=1, max=30, step=1
                ),
                html.Button(
                    "Predict",
                    id="predict-button",
                    n_clicks=0,
                    style={
                        "backgroundColor": "#856404",
                        "color": "white",
                        "border": "none",
                        "padding": "10px 20px",
                        "margin": "10px",
                        "borderRadius": "5px",
                    },
                ),
            ],
            style={"margin": "20px", "textAlign": "center"},
        ),
        dcc.Loading(
            id="loading",
            type="circle",
            children=[
                html.Div(id="prediction-output"),
                dcc.Graph(id="prediction-graph"),
                html.Div(id="prediction-table-container"),
            ],
        ),
    ],
    style={"fontFamily": "Arial", "margin": "0 auto", "maxWidth": "1200px"},
)


@app.callback(
    [
        Output("prediction-graph", "figure"),
        Output("prediction-output", "children"),
        Output("prediction-table-container", "children"),
    ],
    [Input("predict-button", "n_clicks")],
    [Input("days-input", "value")],
)
def update_prediction(n_clicks, days_to_predict):
    if n_clicks == 0:
        return go.Figure(), "", html.Div()

    try:
        if not os.path.exists(FILE_PATH):
            raise FileNotFoundError(f"CSV file not found at: {FILE_PATH}")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

        df = pd.read_csv(FILE_PATH)


        if pd.api.types.is_datetime64_any_dtype(df["Price"]):
            pass
        else:
            df["Price"] = pd.to_datetime(df["Price"])

        last_date = df["Price"].max()

        future_date = pd.date_range(
            start=last_date + timedelta(days=1), periods=days_to_predict, freq="d"
        )
        future_date = pd.DataFrame({"Price": future_date})

        future_date["year"] = future_date["Price"].dt.year
        future_date["month"] = future_date["Price"].dt.month
        future_date["day"] = future_date["Price"].dt.day
        future_date["dayofweek"] = future_date["Price"].dt.dayofweek
        future_date["quarter"] = future_date["Price"].dt.quarter
        future_date["hour"] = future_date["Price"].dt.hour

        df_pre = pd.concat([df, future_date], ignore_index=True)

        df_pre["Close_lag_1week"] = df_pre["Close"].shift(192)
        df_pre["SMA_7"] = df_pre["Close"].shift(162).rolling(window=192).mean()
        df_pre["Close_rolling_std_7"] = (
            df_pre["Close"].shift(192).rolling(window=192).std()
        )
        df_pre["EMA_5"] = df_pre["Close"].ewm(span=5, adjust=False).mean()
        df_pre["Close_rolling_max_7"] = (
            df_pre["Close"].shift(192).rolling(window=192).max()
        )
        df_pre["Close_rolling_min_7"] = (
            df_pre["Close"].shift(192).rolling(window=192).min()
        )
        df_pre["Price_Change"] = df_pre["Close"].shift(168).diff()
        df_pre["Gain"] = df_pre["Price_Change"].where(df_pre["Price_Change"] > 0, 0)
        df_pre["Loss"] = -df_pre["Price_Change"].where(df_pre["Price_Change"] < 0, 0)
        df_pre["Avg_Gain"] = df_pre["Gain"].rolling(window=336).mean()
        df_pre["Avg_Loss"] = df_pre["Loss"].rolling(window=336).mean()
        df_pre["RS"] = df_pre["Avg_Gain"] / df_pre["Avg_Loss"]
        df_pre["RSI"] = 100 - (100 / (1 + df_pre["RS"]))
        df_pre["EMA_1week"] = df_pre["Close"].ewm(span=168, adjust=False).mean()
        df_pre["EMA_2week"] = df_pre["Close"].ewm(span=336, adjust=False).mean()
        df_pre["MACD"] = df_pre["EMA_1week"] - df_pre["EMA_2week"]
        df_pre["Signal"] = df_pre["MACD"].ewm(span=9, adjust=False).mean()
        df_pre["Histogram"] = df_pre["MACD"] - df_pre["Signal"]
        df_pre["Price_Change"] = df_pre["Close"].shift(168).diff()
        df_pre["Volume_Profit"] = df_pre["Volume"].shift(168) * df_pre["Price_Change"]
        df_pre = df_pre.drop(columns=["Volume", "Close"])
        df_pre.set_index("Price", inplace=True)


        try:
            lightgbm_model_gold = joblib.load(MODEL_PATH)
        except Exception as load_err:
            try:
                lightgbm_model_gold = load_model(MODEL_PATH_PYCARET)
            except Exception as pycaret_err:
                model_dir = os.path.dirname(MODEL_PATH)
                model_name = os.path.basename(MODEL_PATH).replace(".pkl", "")
                try:
                    current_dir = os.getcwd()
                    os.chdir(model_dir)
                    lightgbm_model_gold = load_model(model_name)
                    os.chdir(current_dir)
                except Exception as dir_err:
                    raise Exception(
                        f"Failed to load model. Tried multiple methods.\n"
                        f"Direct load error: {load_err}\n"
                        f"PyCaret error: {pycaret_err}\n"
                        f"Directory method error: {dir_err}"
                    )

        predict = predict_model(lightgbm_model_gold, data=df_pre)

        future_predictions = predict.iloc[-days_to_predict:]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=future_predictions.index,
                y=future_predictions["prediction_label"],
                mode="lines+markers",
                name="Prediction",
                line=dict(color="#B8860B", width=3),
                marker=dict(size=8, symbol="diamond"),
            )
        )

        fig.update_layout(
            title="Gold Price Prediction",
            xaxis_title="Date",
            yaxis_title="Gold Price",
            template="plotly_white",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            height=600,
        )

        prediction_table = html.Div(
            [
                html.H3(
                    "Predicted Gold Prices",
                    style={"textAlign": "center", "marginTop": 30},
                ),
                html.Table(
                    
                    [html.Tr([html.Th("Date"), html.Th("Predicted Price")])] +
                    
                    [
                        html.Tr(
                            [
                                html.Td(
                                    future_predictions.index[i].strftime("%Y-%m-%d")
                                ),
                                html.Td(
                                    f"{future_predictions['prediction_label'].iloc[i]:.2f}"
                                ),
                            ]
                        )
                        for i in range(len(future_predictions))
                    ],
                    style={
                        "margin": "0 auto",
                        "borderCollapse": "collapse",
                        "width": "60%",
                    },
                ),
            ]
        )

        return (
            fig,
            html.H3(
                f"Prediction for the next {days_to_predict} days",
                style={"textAlign": "center", "marginTop": 20},
            ),
            prediction_table,
        )

    except Exception as e:
        
        import traceback

        error_details = traceback.format_exc()

        
        file_info = f"CSV file exists: {os.path.exists(FILE_PATH)}\n"
        file_info += f"Model file exists: {os.path.exists(MODEL_PATH)}"

        return (
            go.Figure(),
            html.Div(
                [
                    html.H3(
                        "Error in prediction",
                        style={"color": "red", "textAlign": "center"},
                    ),
                    html.P(f"Error details: {str(e)}"),
                    html.Pre(
                        file_info,
                        style={
                            "backgroundColor": "#f8f9fa",
                            "padding": "10px",
                            "borderRadius": "5px",
                        },
                    ),
                    html.Pre(
                        error_details,
                        style={
                            "backgroundColor": "#f8f9fa",
                            "padding": "10px",
                            "borderRadius": "5px",
                        },
                    ),
                ]
            ),
            html.Div(),
        )


if __name__ == "__main__":
    app.run_server(debug=True)
