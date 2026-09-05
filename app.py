import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="AI Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🤖 AI Sales Forecasting")
st.sidebar.caption("Sales Prediction Dashboard")

page = st.sidebar.radio(
    "Navigate to",
    [
        "📊 Dashboard Overview",
        "📈 Forecast Analysis",
        "🎯 Model Performance",
        "🔮 Forecast Details",
        "🔍 Interactive Filters"
    ]
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("data/sales_data_12000.csv")

    df["Order_Date"] = pd.to_datetime(df["Order_Date"])

    # Handle missing values
    df["Discount"] = df["Discount"].fillna(0)

    return df


# =========================================================
# DYNAMIC FUTURE FORECAST
# =========================================================

@st.cache_resource
def load_forecasting_model():

    model = joblib.load(
        "models/sales_forecasting_model.pkl"
    )

    return model


@st.cache_data
def load_dynamic_forecast():

    # -----------------------------------------------------
    # LOAD FORECASTING DATA
    # -----------------------------------------------------

    df_forecast = pd.read_csv(
        "data/forecasting_sales_data.csv"
    )

    df_forecast["Month_Date"] = pd.to_datetime(
        df_forecast["Month_Date"]
    )

    # -----------------------------------------------------
    # LOAD TRAINED FORECASTING MODEL
    # -----------------------------------------------------

    model = load_forecasting_model()

    # -----------------------------------------------------
    # SORT DATA
    # -----------------------------------------------------

    df_forecast = (
        df_forecast
        .sort_values(
            ["Category", "Region", "Month_Date"]
        )
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # GET LAST 3 MONTHS FOR EACH CATEGORY + REGION
    # -----------------------------------------------------

    groups = []

    for (category, region), group in df_forecast.groupby(
        ["Category", "Region"]
    ):

        group = group.sort_values("Month_Date")

        if len(group) >= 3:

            groups.append(
                group.tail(3).copy()
            )

    # -----------------------------------------------------
    # GENERATE NEXT 3 MONTHS
    # -----------------------------------------------------

    future_predictions = []

    for group in groups:

        category = group["Category"].iloc[-1]

        region = group["Region"].iloc[-1]

        history = list(
            group["Sales"].values
        )

        last_date = group["Month_Date"].iloc[-1]

        # -----------------------------------------------
        # NEXT 3 MONTHS
        # -----------------------------------------------

        for step in range(1, 4):

            future_date = (
                last_date
                + pd.DateOffset(months=step)
            )

            # -------------------------------------------
            # LAG FEATURES
            # -------------------------------------------

            lag_1 = history[-1]

            lag_2 = history[-2]

            lag_3 = history[-3]

            rolling_average = np.mean(
                history[-3:]
            )

            # -------------------------------------------
            # SALES GROWTH
            # -------------------------------------------

            if lag_2 != 0:

                growth = (
                    (lag_1 - lag_2)
                    / lag_2
                )

            else:

                growth = 0

            # -------------------------------------------
            # TIME FEATURES
            # -------------------------------------------

            year = future_date.year

            month = future_date.month

            quarter = future_date.quarter

            # -------------------------------------------
            # MODEL INPUT
            # -------------------------------------------

            input_data = pd.DataFrame({

                "Sales_Lag_1": [lag_1],

                "Sales_Lag_2": [lag_2],

                "Sales_Lag_3": [lag_3],

                "Rolling_3_Month_Average":
                    [rolling_average],

                "Sales_Growth":
                    [growth],

                "Year":
                    [year],

                "Month":
                    [month],

                "Quarter":
                    [quarter],

                "Category":
                    [category],

                "Region":
                    [region]
            })

            # -------------------------------------------
            # PREDICT
            # -------------------------------------------

            prediction = model.predict(
                input_data
            )[0]

            # Prevent negative sales
            prediction = max(
                0,
                prediction
            )

            # -------------------------------------------
            # STORE RESULT
            # -------------------------------------------

            future_predictions.append({

                "Month_Date":
                    future_date,

                "Category":
                    category,

                "Region":
                    region,

                "Predicted_Sales":
                    prediction
            })

            # -------------------------------------------
            # RECURSIVE FORECASTING
            # -------------------------------------------

            history.append(
                prediction
            )

    # -----------------------------------------------------
    # CREATE FORECAST DATAFRAME
    # -----------------------------------------------------

    forecast_df = pd.DataFrame(
        future_predictions
    )

    return forecast_df

# ---------------------------------------------------
# LOAD TRAINED ML MODEL
# ---------------------------------------------------

@st.cache_resource
def load_trained_model():

    model = joblib.load(
        "models/sales_prediction_model.pkl"
    )

    return model


trained_model = load_trained_model()

# ---------------------------------------------------
# GENERATE PREDICTIONS USING TRAINED MODEL
# ---------------------------------------------------

@st.cache_data
def generate_model_predictions():

    ml_data = pd.read_csv(
        "data/ml_sales_data.csv"
    )

    features = [
        "Quantity",
        "Unit_Price",
        "Discount",
        "Year",
        "Month",
        "Quarter",
        "Day",
        "Day_of_Week_Number",
        "Category",
        "Region"
    ]

    target = "Sales"

    X = ml_data[features]
    y = ml_data[target]

    # Same train/test split used during model training
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    predictions = trained_model.predict(X_test)

    return y_test.reset_index(drop=True), pd.Series(
        predictions
    )


df = load_data()

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

if page == "📊 Dashboard Overview":

    st.title("📊 AI Sales Forecasting Dashboard")

    st.markdown(
        "Interactive sales analytics and AI-based sales forecasting"
    )

    st.divider()

    # ---------------------------------------------------
    # SIDEBAR FILTERS
    # ---------------------------------------------------

    st.sidebar.header("🔎 Filters")

    # Category filter
    categories = sorted(df["Category"].dropna().unique())

    selected_categories = st.sidebar.multiselect(
        "Select Category",
        categories,
        default=categories
    )

    # Region filter
    regions = sorted(df["Region"].dropna().unique())

    selected_regions = st.sidebar.multiselect(
        "Select Region",
        regions,
        default=regions
    )

    # Date filter
    min_date = df["Order_Date"].min().date()
    max_date = df["Order_Date"].max().date()

    selected_dates = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # ---------------------------------------------------
    # APPLY FILTERS
    # ---------------------------------------------------

    filtered_df = df[
        (df["Category"].isin(selected_categories)) &
        (df["Region"].isin(selected_regions))
    ]

    # Date filter
    if len(selected_dates) == 2:

        start_date = pd.to_datetime(selected_dates[0])
        end_date = pd.to_datetime(selected_dates[1])

        filtered_df = filtered_df[
            (filtered_df["Order_Date"] >= start_date) &
            (filtered_df["Order_Date"] <= end_date)
        ]

    # ---------------------------------------------------
    # KPI CALCULATIONS
    # ---------------------------------------------------

    total_sales = filtered_df["Sales"].sum()

    total_profit = filtered_df["Profit"].sum()

    total_quantity = filtered_df["Quantity"].sum()

    total_orders = filtered_df["Order_ID"].nunique()

    average_order_value = (
        total_sales / total_orders
        if total_orders > 0
        else 0)

    profit_margin = (
    filtered_df["Profit"].sum()
    / filtered_df["Sales"].sum()
    ) * 100
    

    row1_col1, row1_col2, row1_col3 = st.columns(3)

    with row1_col1:
        st.metric(
            "💰 Total Sales",
            f"₹{total_sales:,.2f}"
        )

    with row1_col2:
        st.metric(
            "📈 Total Profit",
            f"₹{total_profit:,.2f}"
        )

    with row1_col3:
        st.metric(
            "📊 Profit Margin",
            f"{profit_margin:.2f}%"
        )


    row2_col1, row2_col2, row2_col3 = st.columns(3)

    with row2_col1:
        st.metric(
            "🧾 Total Orders",
            f"{total_orders:,}"
        )

    with row2_col2:
        st.metric(
            "📦 Quantity Sold",
            f"{total_quantity:,}"
        )

    with row2_col3:
        st.metric(
            "🛒 Avg Order Value",
            f"₹{average_order_value:,.2f}"
        )

    # ---------------------------------------------------
    # SALES TREND
    # ---------------------------------------------------

    st.subheader("📈 Sales Trend")

    monthly_sales = (
        filtered_df
        .set_index("Order_Date")
        .resample("ME")["Sales"]
        .sum()
    )

    st.line_chart(monthly_sales)

    # ---------------------------------------------------
    # CATEGORY & REGION ANALYSIS
    # ---------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🏷️ Sales by Category")

        category_sales = (
            filtered_df
            .groupby("Category")["Sales"]
            .sum()
            .sort_values(ascending=False)
        )

        st.bar_chart(category_sales)


    with col2:

        st.subheader("🌎 Sales by Region")

        region_sales = (
            filtered_df
            .groupby("Region")["Sales"]
            .sum()
            .sort_values(ascending=False)
        )

        st.bar_chart(region_sales)

    # ---------------------------------------------------
    # PROFIT ANALYSIS
    # ---------------------------------------------------

    st.subheader("💹 Profit by Category")

    category_profit = (
        filtered_df
        .groupby("Category")["Profit"]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(category_profit)

elif page == "📈 Forecast Analysis":

    # ---------------------------------------------------
    # AI SALES FORECAST
    # ---------------------------------------------------

    st.divider()

    st.subheader("🔮 AI Sales Forecast")

    st.caption(
        "Predicted sales for the upcoming months based on the forecasting model."
    )

    # ---------------------------------------------------
    # LOAD FORECAST DATA
    # ---------------------------------------------------


    forecast_df = load_dynamic_forecast()


    # ------------------------------------------------------------
    # FORECAST FILTERS
    # ------------------------------------------------------------

    forecast_categories = ["All"] + sorted(
        forecast_df["Category"].dropna().unique()
    )

    forecast_regions = ["All"] + sorted(
        forecast_df["Region"].dropna().unique()
    )

    # Default values
    default_forecast_category = "All"
    default_forecast_region = "All"

    selected_forecast_category = st.selectbox(
        "Select Forecast Category",
        forecast_categories,
        index=forecast_categories.index(
            default_forecast_category
        )
    )

    selected_forecast_region = st.selectbox(
        "Select Forecast Region",
        forecast_regions,
        index=forecast_regions.index(
            default_forecast_region
        )
    )
    # ---------------------------------------------------
    # APPLY FORECAST FILTERS
    # ---------------------------------------------------

    filtered_forecast = forecast_df.copy()


    if selected_forecast_category != "All":

        filtered_forecast = filtered_forecast[
            filtered_forecast["Category"]
            == selected_forecast_category
        ]


    if selected_forecast_region != "All":

        filtered_forecast = filtered_forecast[
            filtered_forecast["Region"]
            == selected_forecast_region
        ]


    # ---------------------------------------------------
    # MONTHLY FORECAST CALCULATION
    # ---------------------------------------------------

    monthly_forecast = (
        filtered_forecast
        .groupby("Month_Date")["Predicted_Sales"]
        .sum()
        .sort_index()
    )


    # ---------------------------------------------------
    # FORECAST KPI CARDS
    # ---------------------------------------------------

    total_forecast = monthly_forecast.sum()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)


    # Total forecast
    with kpi1:

        st.metric(
            "🔮 Total Forecast",
            f"₹{total_forecast:,.0f}"
        )


    # Individual monthly forecasts
    for i, (column, label) in enumerate(
        zip(
            [kpi2, kpi3, kpi4],
            ["📅 January", "📅 February", "📅 March"]
        )
    ):

        with column:

            if i < len(monthly_forecast):

                value = monthly_forecast.iloc[i]

                st.metric(
                    label,
                    f"₹{value:,.0f}"
                )

            else:

                st.metric(
                    label,
                    "N/A"
                )

    # ------------------------------------------------------------
    # TOP FORECAST REGION
    # ------------------------------------------------------------

    # Start with complete forecast data
    top_region_df = forecast_df.copy()

    # Only apply the Forecast Category filter
    # Do NOT apply the Forecast Region filter
    if selected_forecast_category != "All":
        top_region_df = top_region_df[
            top_region_df["Category"] == selected_forecast_category
        ]

    # Calculate total forecast for every region
    region_totals = (
        top_region_df
        .groupby("Region")["Predicted_Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    # Find the highest forecasting region
    if not region_totals.empty:

        top_region = region_totals.index[0]
        top_region_sales = region_totals.iloc[0]

    else:

        top_region = "N/A"
        top_region_sales = 0


    # Display KPI
    st.metric(
        "🏆 Top Forecast Region",
        top_region,
        f"₹{top_region_sales:,.0f}"
    )

    # ---------------------------------------------------
    # FORECAST TREND
    # ---------------------------------------------------

    st.write("### 📈 Forecast Trend")

    monthly_forecast_display = monthly_forecast.copy()

    month_order = ["January", "February", "March"]

    monthly_forecast_display.index = pd.CategoricalIndex(
        monthly_forecast_display.index.strftime("%B"),
        categories=month_order,
        ordered=True
    )

    monthly_forecast_display = monthly_forecast_display.sort_index()

    import plotly.express as px

    fig = px.line(
        monthly_forecast_display,
        x=monthly_forecast_display.index,
        y="Predicted_Sales",
        markers=True,
        text="Predicted_Sales"
    )

    fig.update_traces(
        texttemplate="₹%{text:,.0f}",
        textposition="top center",
        line=dict(width=3),
        marker=dict(size=9),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Predicted Sales: ₹%{y:,.0f}"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        xaxis_title="Forecast Month",
        yaxis_title="Predicted Sales",
        height=500,
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )



    # ------------------------------------------------------------
    # FORECAST BY REGION
    # ------------------------------------------------------------

    st.divider()

    st.write("### 🌍 Forecast by Region")

    # Start with complete forecast data
    region_forecast_df = forecast_df.copy()

    # Apply Forecast Category filter
    if selected_forecast_category != "All":
        region_forecast_df = region_forecast_df[
            region_forecast_df["Category"] == selected_forecast_category
        ]

    # Apply Forecast Region filter
    if selected_forecast_region != "All":
        region_forecast_df = region_forecast_df[
            region_forecast_df["Region"] == selected_forecast_region
        ]

    # Calculate forecast sales by region
    forecast_by_region = (
        region_forecast_df
        .groupby("Region")["Predicted_Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    # Display chart
    st.bar_chart(
        forecast_by_region,
        x_label="Region",
        y_label="Predicted Sales",
        use_container_width=True
    )

    # ------------------------------------------------------------
    # MONTHLY FORECAST BY REGION
    # ------------------------------------------------------------

    st.divider()

    st.subheader("📅 Monthly Forecast by Region")

    # Start with forecast data
    monthly_region_df = forecast_df.copy()

    # Apply forecast category filter
    if selected_forecast_category != "All":
        monthly_region_df = monthly_region_df[
            monthly_region_df["Category"] == selected_forecast_category
        ]

    # Apply forecast region filter
    if selected_forecast_region != "All":
        monthly_region_df = monthly_region_df[
            monthly_region_df["Region"] == selected_forecast_region
        ]

    # Create region × month table
    monthly_region_table = (
        monthly_region_df
        .pivot_table(
            index="Region",
            columns=monthly_region_df["Month_Date"].dt.strftime("%B"),
            values="Predicted_Sales",
            aggfunc="sum"
        )
        .fillna(0)
    )

    # Correct month order
    month_order = ["January", "February", "March"]

    monthly_region_table = monthly_region_table.reindex(
        columns=[
            month for month in month_order
            if month in monthly_region_table.columns
        ],
        fill_value=0
    )

    # Add total
    monthly_region_table["Total Forecast"] = (
        monthly_region_table.sum(axis=1)
    )

    # Display
    st.dataframe(
        monthly_region_table.style.format("₹{:,.0f}"),
        use_container_width=True
    )

elif page == "🎯 Model Performance":

    # ============================================================
    # MODEL PERFORMANCE
    # ============================================================

    st.divider()

    st.subheader("📊 Model Performance")

    st.caption(
        "Performance comparison of the machine learning models used for sales prediction."
    )

    # ------------------------------------------------------------
    # MODEL PERFORMANCE DATA
    # ------------------------------------------------------------


    model_results_file = "data/model_results.csv"

    model_results = pd.read_csv(
        model_results_file
    )

    # Reset model names from CSV index column
    if "Unnamed: 0" in model_results.columns:
        model_results = model_results.rename(
            columns={"Unnamed: 0": "Model"}
        )

    # Rename R2 for dashboard display
    if "R2" in model_results.columns:
        model_results = model_results.rename(
            columns={"R2": "R²"}
        )

    # Make sure the model names are strings
    model_results["Model"] = (
        model_results["Model"].astype(str)
    )

    st.dataframe(
        model_results,
        use_container_width=True,
        hide_index=True
    )

    # ------------------------------------------------------------
    # AUTOMATIC BEST MODEL SELECTION
    # ------------------------------------------------------------

    # Select the model with the lowest MAE
    best_model_row = model_results.loc[
        model_results["MAE"].idxmin()
    ]

    best_model = best_model_row["Model"]
    best_model_mae = best_model_row["MAE"]
    best_model_rmse = best_model_row["RMSE"]
    best_model_r2 = best_model_row["R²"]


    st.divider()

    st.subheader("🏆 Best Performing Model")

    st.success(
        f"**{best_model}** is the best performing model based on "
        f"the lowest Mean Absolute Error (MAE)."
    )


    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Best Model",
            best_model
        )

    with col2:
        st.metric(
            "Lowest MAE",
            f"₹{best_model_mae:,.2f}"
        )

    with col3:
        st.metric(
            "R² Score",
            f"{best_model_r2:.4f}"
        )

    # ------------------------------------------------------------
    # MODEL RANKING
    # ------------------------------------------------------------

    st.subheader("📊 Model Ranking")

    model_ranking = (
        model_results
        .sort_values("MAE", ascending=True)
        .reset_index(drop=True)
    )

    model_ranking.insert(
        0,
        "Rank",
        range(1, len(model_ranking) + 1)
    )

    st.dataframe(
        model_ranking,
        use_container_width=True,
        hide_index=True
    )

    st.info(
    f"""
    **Why {best_model}?**

    The model was selected because it achieved the lowest
    Mean Absolute Error (MAE) among the evaluated models.

    Lower MAE means the model's predictions are, on average,
    closer to the actual sales values.
    """
    )

    # ---------------------------------------------------
    # LIVE TRAINED MODEL EVALUATION
    # ---------------------------------------------------

    st.divider()

    st.subheader("🤖 Live Trained Model Evaluation")

    st.caption(
        "Metrics calculated directly using the saved trained model."
    )

    try:

        actual_live, predicted_live = generate_model_predictions()

        live_mae = mean_absolute_error(
            actual_live,
            predicted_live
        )

        live_rmse = np.sqrt(
            mean_squared_error(
                actual_live,
                predicted_live
            )
        )

        live_r2 = r2_score(
            actual_live,
            predicted_live
        )

        live_col1, live_col2, live_col3 = st.columns(3)

        with live_col1:
            st.metric(
                "Live MAE",
                f"₹{live_mae:,.2f}"
            )

        with live_col2:
            st.metric(
                "Live RMSE",
                f"₹{live_rmse:,.2f}"
            )

        with live_col3:
            st.metric(
                "Live R²",
                f"{live_r2:.4f}"
            )

        st.success(
            f"Model loaded successfully: **{type(trained_model.named_steps['model']).__name__}**"
        )

    except Exception as e:

        st.error(
        f"Unable to evaluate the trained model: {e}"
        )


    # ============================================================
    # LOAD ACTUAL VS PREDICTED DATA
    # ============================================================

    st.divider()

    st.subheader("📈 Actual vs Predicted Sales")

    st.caption(
        "Comparison between actual sales and sales predicted by the best forecasting model."
    )

    prediction_file = "data/prediction_results.csv"

    try:

        actual_vs_predicted = pd.read_csv(
            prediction_file
        )

    except FileNotFoundError:

        st.error(
            "Prediction results not found. Please run train_model.py first."
        )

        actual_vs_predicted = pd.DataFrame()


    # ============================================================
    # ONLY CONTINUE IF DATA EXISTS
    # ============================================================

    if not actual_vs_predicted.empty:

        # --------------------------------------------------------
        # ACTUAL VS PREDICTED CHART
        # --------------------------------------------------------

        chart_data = actual_vs_predicted[
            [
                "Actual Sales",
                "Predicted Sales"
            ]
        ].copy()

        st.line_chart(
            chart_data,
            use_container_width=True
        )

        # ========================================================
        # FORECAST ACCURACY
        # ========================================================

        st.divider()

        st.subheader("🎯 Forecast Accuracy")

        st.caption(
            "Accuracy metrics showing how closely the model predictions "
            "match the actual sales."
        )

        # --------------------------------------------------------
        # ACTUAL AND PREDICTED VALUES
        # --------------------------------------------------------

        actual = actual_vs_predicted["Actual Sales"]

        predicted = actual_vs_predicted["Predicted Sales"]

        # --------------------------------------------------------
        # MAE
        # --------------------------------------------------------

        mae = mean_absolute_error(
            actual,
            predicted
        )

        # --------------------------------------------------------
        # RMSE
        # --------------------------------------------------------

        rmse = mean_squared_error(
            actual,
            predicted
        ) ** 0.5

        # --------------------------------------------------------
        # R2 SCORE
        # --------------------------------------------------------

        r2 = r2_score(
            actual,
            predicted
        )

        # --------------------------------------------------------
        # MAPE
        # --------------------------------------------------------

        non_zero_actual = actual != 0

        if non_zero_actual.sum() > 0:

            mape = (
                abs(
                    (
                        actual[non_zero_actual]
                        - predicted[non_zero_actual]
                    )
                    / actual[non_zero_actual]
                ).mean()
                * 100
            )

        else:

            mape = 0

        # --------------------------------------------------------
        # FORECAST ACCURACY
        # --------------------------------------------------------

        accuracy = max(
            0,
            100 - mape
        )

        # ========================================================
        # ACCURACY KPI CARDS
        # ========================================================

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "MAE",
                f"₹{mae:,.2f}"
            )

        with col2:

            st.metric(
                "RMSE",
                f"₹{rmse:,.2f}"
            )

        with col3:

            st.metric(
                "R² Score",
                f"{r2:.4f}"
            )

        with col4:

            st.metric(
                "Forecast Accuracy",
                f"{accuracy:.2f}%"
            )

        # ========================================================
        # PREDICTION ERROR ANALYSIS
        # ========================================================

        st.divider()

        st.subheader("📉 Prediction Error Analysis")

        st.caption(
            "Analysis of the difference between actual sales and "
            "sales predicted by the best forecasting model."
        )

        # --------------------------------------------------------
        # CREATE ERROR ANALYSIS DATA
        # --------------------------------------------------------

        error_analysis = actual_vs_predicted.copy()

        error_analysis["Prediction Error"] = (
            error_analysis["Actual Sales"]
            - error_analysis["Predicted Sales"]
        )

        error_analysis["Absolute Error"] = (
            error_analysis["Prediction Error"]
            .abs()
        )

        # ========================================================
        # ERROR KPI CARDS
        # ========================================================

        col1, col2, col3 = st.columns(3)

        with col1:

            avg_error = (
                error_analysis["Prediction Error"].mean()
            )

            st.metric(
                "Average Prediction Error",
                f"₹{avg_error:,.2f}"
            )

        with col2:

            max_error = (
                error_analysis["Absolute Error"].max()
            )

            st.metric(
                "Maximum Error",
                f"₹{max_error:,.2f}"
            )

        with col3:

            median_error = (
                error_analysis["Absolute Error"].median()
            )

            st.metric(
                "Median Absolute Error",
                f"₹{median_error:,.2f}"
            )

        # ========================================================
        # LARGEST PREDICTION ERRORS
        # ========================================================

        st.write("### 🔍 Largest Prediction Errors")

        top_errors = (
            error_analysis
            .sort_values(
                "Absolute Error",
                ascending=False
            )
            .head(20)
            .copy()
        )

        top_errors = top_errors[
            [
                "Actual Sales",
                "Predicted Sales",
                "Prediction Error",
                "Absolute Error"
            ]
        ]

        st.dataframe(
            top_errors,
            use_container_width=True,
            hide_index=True
        )

        # ========================================================
        # PREDICTION ERROR TREND
        # ========================================================

        st.write("### 📊 Prediction Error Trend")

        error_chart = error_analysis[
            ["Prediction Error"]
        ].copy()

        st.line_chart(
            error_chart,
            use_container_width=True
        )

        # ========================================================
        # ERROR DISTRIBUTION
        # ========================================================

        st.write("### 📊 Error Distribution")

        st.caption(
            "Distribution of prediction errors made by the selected forecasting model."
        )

        error_distribution = error_analysis[
            ["Prediction Error"]
        ].copy()

        st.bar_chart(
            error_distribution,
            use_container_width=True
        )

    else:

        st.warning(
            "Model performance analysis cannot be displayed "
            "because prediction_results.csv is unavailable."
        )


elif page == "🔮 Forecast Details":

    # =========================================================
    # SALES FORECAST SUMMARY
    # =========================================================

    st.divider()

    st.subheader("📊 Sales Forecast Summary")

    st.caption(
        "Summary of expected sales for the upcoming forecast period."
    )

    # Load future forecast data
    forecast_summary_df = load_dynamic_forecast()

    # Aggregate future forecast by month
    forecast_summary_monthly = (
        forecast_summary_df
        .groupby("Month_Date")["Predicted_Sales"]
        .sum()
        .reset_index()
    )

    # Calculate summary values
    total_forecast = (
        forecast_summary_monthly["Predicted_Sales"].sum()
    )

    average_forecast = (
        forecast_summary_monthly["Predicted_Sales"].mean()
    )

    highest_row = forecast_summary_monthly.loc[
        forecast_summary_monthly["Predicted_Sales"].idxmax()
    ]

    lowest_row = forecast_summary_monthly.loc[
        forecast_summary_monthly["Predicted_Sales"].idxmin()
    ]

    highest_month = highest_row["Month_Date"].strftime("%B")
    highest_sales = highest_row["Predicted_Sales"]

    lowest_month = lowest_row["Month_Date"].strftime("%B")
    lowest_sales = lowest_row["Predicted_Sales"]


    # Display KPIs
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Forecast Sales",
            f"₹{total_forecast:,.0f}"
        )

    with col2:
        st.metric(
            "Average Monthly Sales",
            f"₹{average_forecast:,.0f}"
        )

    with col3:
        st.metric(
            "Highest Forecast Month",
            highest_month,
            f"₹{highest_sales:,.0f}"
        )

    with col4:
        st.metric(
            "Lowest Forecast Month",
            lowest_month,
            f"₹{lowest_sales:,.0f}"
        )
    # =========================================================
    # FORECAST INSIGHTS
    # =========================================================

    st.divider()

    st.subheader("💡 Forecast Insights")

    st.caption(
        "Key business insights generated from the future sales forecast."
    )

    # ---------------------------------------------------------
    # Calculate forecast insights
    # ---------------------------------------------------------

    # Highest forecast month
    highest_month_row = forecast_summary_monthly.loc[
        forecast_summary_monthly["Predicted_Sales"].idxmax()
    ]

    # Lowest forecast month
    lowest_month_row = forecast_summary_monthly.loc[
        forecast_summary_monthly["Predicted_Sales"].idxmin()
    ]

    highest_month = highest_month_row["Month_Date"].strftime("%B")
    highest_sales = highest_month_row["Predicted_Sales"]

    lowest_month = lowest_month_row["Month_Date"].strftime("%B")
    lowest_sales = lowest_month_row["Predicted_Sales"]

    total_forecast = forecast_summary_monthly["Predicted_Sales"].sum()

    average_forecast = forecast_summary_monthly["Predicted_Sales"].mean()

    # ---------------------------------------------------------
    # Region forecast
    # ---------------------------------------------------------

    region_forecast = (
        forecast_summary_df
        .groupby("Region")["Predicted_Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    top_region = region_forecast.index[0]
    top_region_sales = region_forecast.iloc[0]

    lowest_region = region_forecast.index[-1]
    lowest_region_sales = region_forecast.iloc[-1]


    # ---------------------------------------------------------
    # Display insights
    # ---------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            f"""
            📈 **Highest Forecast Month**

            **{highest_month}** is expected to generate the highest
            sales of approximately **₹{highest_sales:,.0f}**.
            """
        )

        st.success(
            f"""
            🏆 **Top Forecast Region**

            **{top_region}** has the highest expected sales,
            with approximately **₹{top_region_sales:,.0f}** forecast.
            """
        )

    with col2:

        st.warning(
            f"""
            📉 **Lowest Forecast Month**

            **{lowest_month}** is expected to have the lowest
            sales of approximately **₹{lowest_sales:,.0f}**.
            """
        )

        st.info(
            f"""
            📊 **Lowest Forecast Region**

            **{lowest_region}** has the lowest expected sales,
            with approximately **₹{lowest_region_sales:,.0f}** forecast.
            """
        )


    # ---------------------------------------------------------
    # Overall forecast statement
    # ---------------------------------------------------------

    st.markdown(
        f"""
        ### 🔎 Overall Forecast

        The model predicts approximately **₹{total_forecast:,.0f}**
        in total sales over the upcoming forecast period, with an
        average monthly forecast of **₹{average_forecast:,.0f}**.
        """
    )

    # =========================================================
    # FORECAST BY CATEGORY
    # =========================================================

    st.divider()

    st.subheader("📦 Forecast by Category")

    st.caption(
        "Expected sales across product categories during the forecast period."
    )

    # ---------------------------------------------------------
    # Calculate category-wise forecast
    # ---------------------------------------------------------

    category_forecast = (
        forecast_summary_df
        .groupby("Category")["Predicted_Sales"]
        .sum()
        .reset_index()
        .sort_values(
            "Predicted_Sales",
            ascending=False
        )
    )

    # ---------------------------------------------------------
    # Find top and lowest category
    # ---------------------------------------------------------

    top_category_row = category_forecast.iloc[0]

    lowest_category_row = category_forecast.iloc[-1]

    top_category = top_category_row["Category"]
    top_category_sales = top_category_row["Predicted_Sales"]

    lowest_category = lowest_category_row["Category"]
    lowest_category_sales = lowest_category_row["Predicted_Sales"]


    # ---------------------------------------------------------
    # Display category KPIs
    # ---------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "🏆 Top Forecast Category",
            top_category,
            f"₹{top_category_sales:,.0f}"
        )

    with col2:
        st.metric(
            "📉 Lowest Forecast Category",
            lowest_category,
            f"₹{lowest_category_sales:,.0f}"
        )


    # ---------------------------------------------------------
    # Category forecast chart
    # ---------------------------------------------------------

    st.write("### Category-wise Expected Sales")

    st.bar_chart(
        category_forecast.set_index("Category")["Predicted_Sales"],
        use_container_width=True
    )

    # =========================================================
    # FORECAST BY REGION
    # =========================================================

    st.divider()

    st.subheader("🌍 Forecast by Region")

    st.caption(
        "Expected sales across regions during the forecast period."
    )

    # Load detailed forecast data
    region_forecast_df = forecast_summary_df.copy()

    # Group predicted sales by region
    region_forecast = (
        region_forecast_df
        .groupby("Region")["Predicted_Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    # Top and lowest regions
    top_region = region_forecast.idxmax()
    top_region_sales = region_forecast.max()

    lowest_region = region_forecast.idxmin()
    lowest_region_sales = region_forecast.min()

    # KPI cards
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "🏆 Top Forecast Region",
            top_region,
            f"₹{top_region_sales:,.0f}"
        )

    with col2:
        st.metric(
            "📉 Lowest Forecast Region",
            lowest_region,
            f"₹{lowest_region_sales:,.0f}"
        )

    # Chart
    st.write("### Region-wise Expected Sales")

    st.bar_chart(
        region_forecast,
        use_container_width=True
    )

    # =========================================================
    # MONTHLY FORECAST TREND
    # =========================================================

    st.divider()

    st.subheader("📅 Monthly Forecast Trend")

    st.caption(
        "Expected sales and month-to-month changes during the forecast period."
    )

    # Load future forecast data
    monthly_forecast = pd.read_csv(
        "data/future_sales_forecast.csv"
    )

    # Convert date column
    monthly_forecast["Month_Date"] = pd.to_datetime(
        monthly_forecast["Month_Date"]
    )

    # Aggregate forecast by month
    monthly_forecast = (
        monthly_forecast
        .groupby("Month_Date")["Predicted_Sales"]
        .sum()
        .reset_index()
        .sort_values("Month_Date")
    )

    # Calculate month-to-month change
    monthly_forecast["Sales_Change"] = (
        monthly_forecast["Predicted_Sales"].diff()
    )

    # Calculate growth percentage
    monthly_forecast["Growth_Percentage"] = (
        monthly_forecast["Predicted_Sales"]
        .pct_change()
        .fillna(0)
        * 100
    )

    # ---------------------------------------------------------
    # Chart
    # ---------------------------------------------------------

    st.write("### Expected Sales by Month")

    st.line_chart(
        monthly_forecast.set_index("Month_Date")[
            "Predicted_Sales"
        ],
        use_container_width=True
    )

    # ---------------------------------------------------------
    # Monthly details
    # ---------------------------------------------------------

    st.write("### Monthly Forecast Details")

    display_monthly = monthly_forecast.copy()

    display_monthly["Month"] = (
        display_monthly["Month_Date"]
        .dt.strftime("%B %Y")
    )

    display_monthly["Predicted Sales"] = (
        display_monthly["Predicted_Sales"]
        .round(0)
    )

    display_monthly["Growth"] = (
        display_monthly["Growth_Percentage"]
        .round(2)
        .astype(str)
        + "%"
    )

    display_monthly["Sales Change"] = (
        display_monthly["Sales_Change"]
        .round(0)
    )

    st.dataframe(
        display_monthly[
            [
                "Month",
                "Predicted Sales",
                "Sales Change",
                "Growth"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

elif page == "🔍 Interactive Filters":

    # =========================================================
    # INTERACTIVE FORECAST FILTERS
    # =========================================================

    st.divider()

    st.subheader("🎛️ Interactive Forecast Filters")

    st.caption(
        "Filter the forecast by category, region, and forecast month."
    )

    # Load forecast data
    filter_forecast = load_dynamic_forecast()

    # ---------------------------------------------------------
    # FILTER OPTIONS
    # ---------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_category = st.selectbox(
            "Product Category",
            ["All"] + sorted(
                filter_forecast["Category"]
                .dropna()
                .unique()
                .tolist()
            )
        )

    with col2:
        selected_region = st.selectbox(
            "Region",
            ["All"] + sorted(
                filter_forecast["Region"]
                .dropna()
                .unique()
                .tolist()
            )
        )

    with col3:
        selected_month = st.selectbox(
            "Forecast Month",
            ["All"] + sorted(
                filter_forecast["Month_Date"]
                .dt.strftime("%B %Y")
                .unique()
                .tolist()
            )
        )

    # ---------------------------------------------------------
    # APPLY FILTERS
    # ---------------------------------------------------------

    filtered_forecast = filter_forecast.copy()

    if selected_category != "All":
        filtered_forecast = filtered_forecast[
            filtered_forecast["Category"]
            == selected_category
        ]

    if selected_region != "All":
        filtered_forecast = filtered_forecast[
            filtered_forecast["Region"]
            == selected_region
        ]

    if selected_month != "All":
        filtered_forecast = filtered_forecast[
            filtered_forecast["Month_Date"]
            .dt.strftime("%B %Y")
            == selected_month
        ]

    # ---------------------------------------------------------
    # FILTERED RESULTS
    # ---------------------------------------------------------

    st.write("### Filtered Forecast")

    total_filtered_sales = (
        filtered_forecast["Predicted_Sales"].sum()
    )

    average_filtered_sales = (
        filtered_forecast["Predicted_Sales"].mean()
        if len(filtered_forecast) > 0
        else 0
    )

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric(
            "Total Forecast Sales",
            f"₹{total_filtered_sales:,.0f}"
        )

    with result_col2:
        st.metric(
            "Average Forecast Sales",
            f"₹{average_filtered_sales:,.0f}"
        )

    with result_col3:
        st.metric(
            "Forecast Records",
            f"{len(filtered_forecast):,}"
        )

    # ---------------------------------------------------------
    # FILTERED TABLE
    # ---------------------------------------------------------

    if len(filtered_forecast) > 0:

        display_filtered = filtered_forecast.copy()

        display_filtered["Month"] = (
            display_filtered["Month_Date"]
            .dt.strftime("%B %Y")
        )

        display_filtered["Predicted Sales"] = (
            display_filtered["Predicted_Sales"]
            .round(0)
        )

        st.dataframe(
            display_filtered[
                [
                    "Month",
                    "Category",
                    "Region",
                    "Predicted Sales"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "No forecast data is available for the selected filters."
        )



    # ---------------------------------------------------
    # HISTORICAL SALES + FUTURE FORECAST
    # ---------------------------------------------------

    st.divider()

    st.subheader("📈 Historical Sales & Future Forecast")

    st.caption(
        "Historical sales from 2023–2025 followed by AI-generated "
        "sales forecasts for January–March 2026."
    )


    # ---------------------------------------------------
    # FILTER HISTORICAL DATA
    # ---------------------------------------------------

    historical_filtered = df.copy()

    if selected_category != "All":
        historical_filtered = historical_filtered[
            historical_filtered["Category"]
            == selected_category
        ]

    if selected_region != "All":
        historical_filtered = historical_filtered[
            historical_filtered["Region"]
            == selected_region
        ]


    # ---------------------------------------------------
    # HISTORICAL MONTHLY SALES
    # ---------------------------------------------------

    historical_monthly = (
        historical_filtered
        .set_index("Order_Date")
        .resample("MS")["Sales"]
        .sum()
        .sort_index()
    )


    # ---------------------------------------------------
    # FUTURE FORECAST MONTHLY SALES
    # ---------------------------------------------------

    future_monthly = (
        filtered_forecast
        .groupby("Month_Date")["Predicted_Sales"]
        .sum()
        .sort_index()
    )


    # ---------------------------------------------------
    # RENAME SERIES
    # ---------------------------------------------------

    historical_chart = historical_monthly.rename(
        "Historical Sales"
    )

    forecast_chart = future_monthly.rename(
        "Forecast Sales"
    )


    # ---------------------------------------------------
    # COMBINE HISTORICAL + FORECAST
    # ---------------------------------------------------

    combined_chart = pd.concat(
        [
            historical_chart,
            forecast_chart
        ],
        axis=1
    )


    # ---------------------------------------------------
    # DISPLAY CHART
    # ---------------------------------------------------

    st.line_chart(
        combined_chart,
        x_label="Month",
        y_label="Sales",
        use_container_width=True
    )

    # ---------------------------------------------------
    # FORECAST BY CATEGORY
    # ---------------------------------------------------

    st.divider()

    st.subheader("📊 Forecast by Category")

    category_forecast = (
        filtered_forecast
        .groupby(["Category", "Month_Date"])["Predicted_Sales"]
        .sum()
        .reset_index()
    )

    category_forecast["Month"] = category_forecast["Month_Date"].dt.strftime("%B")

    category_table = category_forecast.pivot(
        index="Category",
        columns="Month",
        values="Predicted_Sales"
    )

    month_order = ["January", "February", "March"]

    category_table = category_table.reindex(
        columns=[m for m in month_order if m in category_table.columns]
    )

    category_table["Total Forecast"] = category_table.sum(axis=1)

    st.dataframe(
        category_table.style.format("₹{:,.0f}"),
        use_container_width=True
    )


    # ---------------------------------------------------
    # FORECAST DETAILS
    # ---------------------------------------------------

    st.write("### 📋 Forecast Details")

    display_forecast = filtered_forecast.copy()

    display_forecast["Month_Date"] = (
        display_forecast["Month_Date"]
        .dt.strftime("%Y-%m-%d")
    )

    display_forecast["Predicted_Sales"] = (
        display_forecast["Predicted_Sales"]
        .round(2)
    )

    st.dataframe(
        display_forecast,
        use_container_width=True,
        hide_index=True
    )


    # ---------------------------------------------------
    # DOWNLOAD FORECAST
    # ---------------------------------------------------

    forecast_csv = filtered_forecast.to_csv(
        index=False
    )

    st.download_button(
        label="⬇️ Download Forecast",
        data=forecast_csv,
        file_name="sales_forecast.csv",
        mime="text/csv"
    )

    # ---------------------------------------------------
    # DATA TABLE
    # ---------------------------------------------------

    st.subheader("📋 Sales Data")

    st.dataframe(
        df,
        use_container_width=True,
        height=400
    )

    # ---------------------------------------------------
    # DOWNLOAD BUTTON
    # ---------------------------------------------------

    csv = filtered_forecast.to_csv(index=False)

    st.download_button(
        label="⬇️ Download Filtered Data",
        data=csv,
        file_name="filtered_sales_data.csv",
        mime="text/csv"
    )