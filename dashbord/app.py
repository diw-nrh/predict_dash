import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objs as go
from pycaret.regression import load_model, predict_model
from datetime import datetime, timedelta

app = Dash(__name__)

loaded_model = load_model("D:/Term Pr/website_pm25/pm25_Extra_Trees_Regressor")

LOCATIONS = {
    "bansai": {
        "name": "Bansai",
        "file": "bansai.predict.h.csv",
        "lat": 7.0086,
        "lon": 100.4746,
    },
    "chauatschool": {
        "name": "โรงเรียนชะอวด นครศรี",
        "file": "chauatschool.predict.h.csv",
        "lat": 7.9355,
        "lon": 99.9863,
    },
    "r202": {
        "name": "คณะวิศวกรรม มอ.",
        "file": "r202.predict.h.csv",
        "lat": 7.0095,
        "lon": 100.4982,
    },
    "jsps001": {
        "name": "เทศบาลนครสุราษฎร์ธานี",
        "file": "jsps001.predict.h.csv",
        "lat": 9.1371,
        "lon": 99.3304,
    },
    "jsps013": {
        "name": "สถานีเทศบาลนครหาดใหญ่ 01",
        "file": "jsps013.predict.h.csv",
        "lat": 7.0086,
        "lon": 100.4746,
    },
    "jsps014": {
        "name": "สถานีเทศบาลนครหาดใหญ่ 02",
        "file": "jsps014.predict.h.csv",
        "lat": 7.0086,
        "lon": 100.4746,
    },
}

weather_data = pd.read_csv("D:/Term Pr/website_pm25/bansai.predict.h.csv")


future_dates = pd.date_range(start=pd.to_datetime("today"), periods=7)

future_data = pd.DataFrame(
    {
        "humidity": weather_data["humidity"].iloc[-7:].values,
        "temperature": weather_data["temperature"].iloc[-7:].values,
        "month": future_dates.month,
        "dayofweek": future_dates.dayofweek,
        "weekofyear": future_dates.isocalendar().week,
        "day": future_dates.day,
        "hour": [12] * 7,
        "pm_2_5_lag_12day": [2.830508] * 7,
        "PM2.5_MA3_prev_12day": [6.876808] * 7,
        "PM2.5_MAX24_prev_12day": [34.87931] * 7,
        "PM2.5_MIN24_prev_12day": [0.78] * 7,
    }
)


predictions = predict_model(loaded_model, data=future_data.copy())
predicted_values = predictions["prediction_label"]

historical_dates = pd.date_range(
    start="2024-01-01", end=pd.to_datetime("today") - pd.Timedelta(days=1)
)
historical_data = pd.DataFrame(
    {
        "humidity": [66.01123957554452] * len(historical_dates),
        "temperature": [32.750086008329966] * len(historical_dates),
        "month": historical_dates.month,
        "dayofweek": historical_dates.dayofweek,
        "weekofyear": historical_dates.isocalendar().week,
        "day": historical_dates.day,
        "hour": [12] * len(historical_dates),
        "pm_2_5_lag_12day": [2.830508] * len(historical_dates),
        "PM2.5_MA3_prev_12day": [6.876808] * len(historical_dates),
        "PM2.5_MAX24_prev_12day": [34.87931] * len(historical_dates),
        "PM2.5_MIN24_prev_12day": [0.78] * len(historical_dates),
    }
)

historical_predictions = predict_model(loaded_model, data=historical_data.copy())
historical_values = historical_predictions["prediction_label"]


all_dates = pd.concat([pd.Series(historical_dates), pd.Series(future_dates)])
all_values = pd.concat([historical_values, predicted_values])

mapbox_token = "your_mapbox_token_here"

app.layout = html.Div(
    [
        
        html.Div(
            [
                html.H1("PM2.5 Prediction Dashboard", className="main-title"),
                html.Div(
                    [
                        html.I(className="fas fa-calendar-alt calendar-icon"),
                        html.Span(
                            datetime.now().strftime("%d/%m/%Y"),
                            className="current-date",
                        ),
                    ],
                    className="date-display",
                ),
            ],
            className="header-section",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3("Current Status", className="panel-title"),
                                html.Div(
                                    id="current-day-details",
                                    className="current-day-box",
                                ),
                            ],
                            className="status-panel",
                        ),
                        html.Div(
                            [
                                html.H3("Select Date", className="panel-title"),
                                dcc.DatePickerSingle(
                                    id="date-picker",
                                    min_date_allowed=historical_dates[0],
                                    max_date_allowed=future_dates[-1],
                                    initial_visible_month=datetime.now(),
                                    date=datetime.now().date(),
                                    display_format="DD/MM/YYYY",
                                    className="date-picker",
                                ),
                                html.Div(
                                    id="selected-date-output", className="selected-date"
                                ),
                                html.Div(id="date-details", className="details-box"),
                            ],
                            className="control-panel",
                        ),
                        html.Div(
                            [
                                html.H3("เลือกสถานที่", className="panel-title"),
                                dcc.Dropdown(
                                    id="location-dropdown",
                                    options=[
                                        {"label": info["name"], "value": loc_id}
                                        for loc_id, info in LOCATIONS.items()
                                    ],
                                    value="bansai",  
                                    className="location-dropdown",
                                ),
                            ],
                            className="location-selector",
                        ),
                        html.Div(
                            [
                                html.H3("Location Map", className="panel-title"),
                                dcc.Graph(id="location-map", className="location-map"),
                            ],
                            className="map-panel",
                        ),
                    ],
                    className="left-panel",
                ),
                html.Div(
                    [dcc.Graph(id="pm25-graph", className="main-graph")],
                    className="right-panel",
                ),
            ],
            className="main-content",
        ),
        html.Footer(
            [
                html.P("PM2.5 Quality Levels:", className="footer-title"),
                html.Div(
                    [
                        html.Span("Good (0-25)", className="quality-indicator good"),
                        html.Span(
                            "Moderate (26-50)", className="quality-indicator moderate"
                        ),
                        html.Span(
                            "Unhealthy for Sensitive Groups (51-100)",
                            className="quality-indicator sensitive",
                        ),
                        html.Span(
                            "Unhealthy (>100)", className="quality-indicator unhealthy"
                        ),
                    ],
                    className="quality-indicators",
                ),
            ],
            className="footer-section",
        ),
    ],
    className="dashboard-container",
)

@app.callback(
    Output("current-day-details", "children"),
    Input("pm25-graph", "figure"),
)
def update_current_day(figure): 
    today = datetime.now().date()
    today_idx = (all_dates.dt.date == today).idxmax()

    if today_idx in all_values.index:
        pm25_value = all_values[today_idx]
        quality_level = (
            "ดี"
            if pm25_value <= 25
            else (
                "ปานกลาง"
                if pm25_value <= 50
                else "มีผลกระทบต่อกลุ่มเสี่ยง" if pm25_value <= 100 else "มีผลกระทบต่อสุขภาพ"
            )
        )

        return html.Div(
            [
                html.H4("ค่า PM2.5 วันนี้"),
                html.P(f"{pm25_value:.2f} µg/m³", className="current-value"),
                html.P(f"สถานะ: {quality_level}", className="current-status"),
            ]
        )
    return html.Div("ไม่พบข้อมูลวันนี้")


def generate_predictions(location_id, selected_date):
    """สร้างข้อมูลทำนายสำหรับสถานที่และวันที่ที่เลือก"""
    weather_data = pd.read_csv(
        f"D:/Term Pr/website_pm25/{LOCATIONS[location_id]['file']}"
    )

    future_dates = pd.date_range(start=pd.to_datetime("today"), periods=7)
    future_data = pd.DataFrame(
        {
            "humidity": weather_data["humidity"].iloc[-7:].values,
            "temperature": weather_data["temperature"].iloc[-7:].values,
            "month": future_dates.month,
            "dayofweek": future_dates.dayofweek,
            "weekofyear": future_dates.isocalendar().week,
            "day": future_dates.day,
            "hour": [12] * 7,
            "pm_2_5_lag_12day": [2.830508] * 7,
            "PM2.5_MA3_prev_12day": [6.876808] * 7,
            "PM2.5_MAX24_prev_12day": [34.87931] * 7,
            "PM2.5_MIN24_prev_12day": [0.78] * 7,
        }
    )

    historical_dates = pd.date_range(
        start="2024-01-01", end=pd.to_datetime("today") - pd.Timedelta(days=1)
    )
    historical_data = pd.DataFrame(
        {
            "humidity": [weather_data["humidity"].mean()] * len(historical_dates),
            "temperature": [weather_data["temperature"].mean()] * len(historical_dates),
            "month": historical_dates.month,
            "dayofweek": historical_dates.dayofweek,
            "weekofyear": historical_dates.isocalendar().week,
            "day": historical_dates.day,
            "hour": [12] * len(historical_dates),
            "pm_2_5_lag_12day": [2.830508] * len(historical_dates),
            "PM2.5_MA3_prev_12day": [6.876808] * len(historical_dates),
            "PM2.5_MAX24_prev_12day": [34.87931] * len(historical_dates),
            "PM2.5_MIN24_prev_12day": [0.78] * len(historical_dates),
        }
    )

    future_pred = predict_model(loaded_model, data=future_data)
    hist_pred = predict_model(loaded_model, data=historical_data)

   
    all_dates = pd.concat([pd.Series(historical_dates), pd.Series(future_dates)])
    all_values = pd.concat(
        [hist_pred["prediction_label"], future_pred["prediction_label"]]
    )

    return pd.DataFrame({"date": all_dates, "value": all_values.values})


@app.callback(
    [
        Output("pm25-graph", "figure"),
        Output("selected-date-output", "children"),
        Output("date-details", "children"),
    ],
    [Input("date-picker", "date"), Input("location-dropdown", "value")],
)
def update_graph_and_details(selected_date, selected_location):
    if selected_date is None:
        selected_date = datetime.now().date()
    else:
        selected_date = datetime.strptime(
            selected_date.split("T")[0], "%Y-%m-%d"
        ).date()

    df_display = generate_predictions(selected_location, selected_date)

    center_date = pd.to_datetime(selected_date)
    start_date = center_date - pd.Timedelta(days=3)
    end_date = center_date + pd.Timedelta(days=3)

    mask = (df_display["date"] >= start_date) & (df_display["date"] <= end_date)
    df_filtered = df_display[mask]

    fig = {
        "data": [
            go.Scatter(
                x=df_filtered["date"],
                y=df_filtered["value"],
                mode="lines+markers",
                name="PM2.5",
                line=dict(color="#FFD700", width=2), 
                marker=dict(size=8),
            ),
    
            go.Scatter(
                x=[pd.to_datetime(selected_date)],
                y=[
                    df_filtered[df_filtered["date"].dt.date == selected_date][
                        "value"
                    ].iloc[0]
                ],
                mode="markers",
                name="Selected Date",
                marker=dict(color="#FF4500", size=15, symbol="diamond"),
                hovertemplate="<b>Selected Date</b><br>"
                + "Date: %{x|%d/%m/%Y}<br>"
                + "PM2.5: %{y:.2f} µg/m³<extra></extra>",
            ),
        ],
        "layout": go.Layout(
            title=f"PM2.5 Values at {LOCATIONS[selected_location]['name']}",
            xaxis={
                "title": "Date",
                "showgrid": True,
                "gridcolor": "lightgray",
                "tickformat": "%d/%m/%Y",
                "range": [start_date, end_date],
            },
            yaxis={
                "title": "PM2.5 (µg/m³)",
                "showgrid": True,
                "gridcolor": "lightgray",
            },
            plot_bgcolor="#FFFFFF",  
            paper_bgcolor="#FFFFFF", 
            hovermode="x unified",
            height=450,
            margin=dict(l=40, r=20, t=40, b=40),
        ),
    }

    selected_value = df_filtered[
        pd.to_datetime(df_filtered["date"]).dt.date == selected_date
    ]["value"].iloc[0]
    if not pd.isna(selected_value):
        quality_level = (
            "Good"
            if selected_value <= 25
            else (
                "Moderate"
                if selected_value <= 50
                else (
                    "Unhealthy for Sensitive Groups"
                    if selected_value <= 100
                    else "Unhealthy"
                )
            )
        )

        date_details = html.Div(
            [
                html.H4(f"PM2.5 Data for {selected_date.strftime('%d/%m/%Y')}"),
                html.P(f"Value: {selected_value:.2f} µg/m³"),
                html.P(f"Quality: {quality_level}"),
            ]
        )
        date_output = f"Selected: {selected_date.strftime('%d/%m/%Y')}"
    else:
        date_details = html.Div([html.H4("No data available")])
        date_output = "Please select a date"

    return fig, date_output, date_details


@app.callback(
    Output("location-map", "figure"),
    [Input("location-dropdown", "value"), Input("date-picker", "date")],
)
def update_map(selected_location, selected_date):
    if selected_date is None:
        selected_date = datetime.now().date()
    else:
        selected_date = datetime.strptime(
            selected_date.split("T")[0], "%Y-%m-%d"
        ).date()

    pm25_values = {}
    for loc_id in LOCATIONS.keys():
        try:
            df = generate_predictions(loc_id, selected_date)
            selected_value = df[pd.to_datetime(df["date"]).dt.date == selected_date][
                "value"
            ].iloc[0]
            pm25_values[loc_id] = selected_value
        except Exception as e:
            print(f"Error processing data for {loc_id}: {str(e)}")
            pm25_values[loc_id] = 0

    def get_marker_color(value):
        if value <= 25:
            return "#8A2BE2" 
        elif value <= 50:
            return "#FFD700" 
        elif value <= 100:
            return "#FF4500" 
        else:
            return "#DC3545"  

 
    def get_status_text(value):
        if value <= 25:
            return "ดี"
        elif value <= 50:
            return "ปานกลาง"
        elif value <= 100:
            return "เสี่ยง"
        else:
            return "อันตราย"

    return {
        "data": [
            {
                "type": "scattermapbox",
                "lat": [info["lat"] for info in LOCATIONS.values()],
                "lon": [info["lon"] for info in LOCATIONS.values()],
                "mode": "markers+text",
                "marker": {
                    "size": 15,
                    "color": [
                        (
                            "#FF0000"
                            if loc == selected_location
                            else get_marker_color(pm25_values[loc])
                        )
                        for loc in LOCATIONS.keys()
                    ],
                    "opacity": 0.8,
                },
                "text": [
                    f"{info['name']}<br>"
                    f"PM2.5: {pm25_values[loc_id]:.2f} µg/m³<br>"
                    f"สถานะ: {get_status_text(pm25_values[loc_id])}<br>"
                    f"วันที่: {selected_date.strftime('%d/%m/%Y')}"
                    for loc_id, info in LOCATIONS.items()
                ],
                "textposition": "top center",
                "hoverinfo": "text",
            }
        ],
        "layout": {
            "mapbox": {
                "style": "open-street-map",
                "center": {"lat": 8.5, "lon": 100.0},
                "zoom": 6.5,
                "pitch": 0,
                "bearing": 0,
            },
            "autosize": True,
            "height": 500,
            "margin": {"r": 0, "t": 0, "l": 0, "b": 0},
            "showlegend": False,
            "uirevision": True,
            "dragmode": "zoom",  
            "scrollZoom": True, 
            "modebar": {
                "remove": [],
                "add": ["zoomInMapbox", "zoomOutMapbox", "resetViewMapbox"],
                "bgcolor": "rgba(30, 30, 30, 0.9)",  
                "color": "#FFD700",  
                "activecolor": "#ff7f0e",
                "orientation": "h",
            },
        },
    }

if __name__ == "__main__":
    app.run_server(debug=True)
