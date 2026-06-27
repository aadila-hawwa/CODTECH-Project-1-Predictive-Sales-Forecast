import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(page_title="Predictive Sales Forecast Dashboard", layout="wide")

df = pd.read_csv("cleaned_forecast_data.zip",compression="zip")
forecast_df = pd.read_csv("forecast_model_data.csv")

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
forecast_df["InvoiceDate"] = pd.to_datetime(forecast_df["InvoiceDate"])

model = joblib.load("sales_forecast_model.pkl")

st.title("Predictive Sales Forecast Dashboard")

total_sales = df["Sales"].sum()
average_sales = df["Sales"].mean()
total_orders = df["InvoiceNo"].nunique()
total_customers = df["CustomerID"].nunique()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Sales", f"{total_sales:,.2f}")
col2.metric("Average Sales", f"{average_sales:,.2f}")
col3.metric("Total Orders", total_orders)
col4.metric("Total Customers", total_customers)

tab1, tab2, tab3, tab4 = st.tabs([
    "Sales Trend",
    "Business Analysis",
    "Forecast",
    "Data Preview"
])

with tab1:
    monthly_sales = (
        df.resample("ME", on="InvoiceDate")["Sales"]
        .sum()
        .reset_index()
    )

    fig1 = px.line(
        monthly_sales,
        x="InvoiceDate",
        y="Sales",
        title="Monthly Sales Trend",
        markers=True
    )

    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    country_sales = (
        df.groupby("Country")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig2 = px.bar(
        country_sales,
        x="Country",
        y="Sales",
        title="Top 10 Countries by Sales"
    )

    st.plotly_chart(fig2, use_container_width=True)

    product_sales = (
        df.groupby("Description")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig3 = px.bar(
        product_sales,
        x="Sales",
        y="Description",
        orientation="h",
        title="Top 10 Products by Sales"
    )

    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    st.subheader("Next 7 Days Sales Forecast")

    last_data = forecast_df.copy()
    future_predictions = []

    for i in range(7):
        last_date = last_data["InvoiceDate"].max()
        next_date = last_date + pd.Timedelta(days=1)

        lag_1 = last_data.iloc[-1]["Sales"]
        lag_7 = last_data.iloc[-7]["Sales"]
        rolling_7 = last_data.tail(7)["Sales"].mean()

        future_row = pd.DataFrame({
            "Year": [next_date.year],
            "Month": [next_date.month],
            "Day": [next_date.day],
            "DayOfWeek": [next_date.dayofweek],
            "WeekOfYear": [next_date.isocalendar().week],
            "Lag_1": [lag_1],
            "Lag_7": [lag_7],
            "Rolling_7": [rolling_7]
        })

        predicted_sales = model.predict(future_row)[0]

        future_predictions.append({
            "Date": next_date,
            "Predicted Sales": predicted_sales
        })

        new_row = {
            "InvoiceDate": next_date,
            "Sales": predicted_sales,
            "Year": next_date.year,
            "Month": next_date.month,
            "Day": next_date.day,
            "DayOfWeek": next_date.dayofweek,
            "WeekOfYear": next_date.isocalendar().week,
            "Lag_1": lag_1,
            "Lag_7": lag_7,
            "Rolling_7": rolling_7
        }

        last_data = pd.concat([last_data, pd.DataFrame([new_row])], ignore_index=True)

    future_df = pd.DataFrame(future_predictions)

    fig4 = px.line(
        future_df,
        x="Date",
        y="Predicted Sales",
        markers=True,
        title="Next 7 Days Predicted Sales"
    )

    st.plotly_chart(fig4, use_container_width=True)

    st.dataframe(future_df, use_container_width=True)

with tab4:
    st.dataframe(df, use_container_width=True)
