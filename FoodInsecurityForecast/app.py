from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import plotly.express as px
import pickle
import os

app = Flask(__name__)

# Load the model once at startup
MODEL_PATH = "best_linear_model.pickle"
model = pickle.load(open(MODEL_PATH, "rb"))
predicted_dataframe = None

# Choropleth visualization function
def choropleth(df, value_col, color_scale, title):
    df['FIPS'] = df['FIPS'].astype(str).str.zfill(5)
    fig = px.choropleth(
        df,
        geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
        locations="FIPS",
        color=value_col,
        color_continuous_scale=color_scale,
        scope="usa",
        hover_data=["State", "County", value_col]
    )
    fig.update_layout(title_text=title)
    return fig.to_html(full_html=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get uploaded file
    file = request.files['datafile']
    if not file:
        return "No file uploaded", 400

    df = pd.read_csv(file)

    # Prepare the data for prediction
    X, df_clean = prepareData(df)
    # Predict
    df_clean['Predicted FI Rate'] = model.predict(X)

    # Generate map
    map_html = choropleth(df_clean, 'Predicted FI Rate', 'twilight', 'Predicted Food Insecurity Rates')

    # Save results temporarily
    results_path = "static/results.csv"
    df_clean.to_csv(results_path, index=False)

    selected_cols = ['FIPS','County', 'FI Rate','Predicted FI Rate']
    df_selected = df_clean[selected_cols].copy()

    table_html = df_selected.head(10).to_html(classes='data')
    return render_template(
        'results.html',
        table_html=table_html,
        map_html=map_html
    )


@app.route('/predict_scenario', methods=['POST'])
def predict_scenario():
    df = pd.read_csv("engineered_data.csv")
    TEST_YEAR = 2018
    TRAIN_START = 2010
    df["FIPS"] = df["FIPS"].astype(str).str.zfill(5)
    df["STATE"] = df["FIPS"].str[:2]
    df["Year"] = df["Year"].astype(int)

    # Only rows with a labeled target
    df = df[~df["FI Rate"].isna()].copy()

    # Build lag feature
    df = df.sort_values(["FIPS", "Year"])
    df["FI_prev"] = df.groupby("FIPS")["FI Rate"].shift(1)

    # User inputs
    year = int(request.form['year'])
    unemployment_change = float(request.form['unemployment_change']) / 100
    cost_change = float(request.form['cost_change']) / 100

    # Create a copy of baseline data
    df_future = df[df["Year"] == TEST_YEAR].copy()

    # Apply percentage changes
    if 'Unemployment_rate' in df_future.columns:
        print("inUnemployment_rate")
        df_future['Unemployment_rate'] *= (1 + unemployment_change)
    if 'Cost Per Meal' in df_future.columns:
        print("Cost Per Meal") 
        df_future['Cost Per Meal'] *= (1 + cost_change)

    df_future['Year'] = year

    train = df[(df["Year"] >= TRAIN_START) & (df["Year"] < TEST_YEAR)].copy()

    # === Identify features ===
    IGNORE = {"FIPS", "STATE", "Year", "FI Rate"}
    num_all = [c for c in df.columns if c not in IGNORE and df[c].dtype != "O"]
    cat_all = [c for c in df.columns if c not in IGNORE and df[c].dtype == "O"]

    # Drop sparse (>60% missing)
    DROP_SPARSE = 0.60
    miss = train[num_all + cat_all].isna().mean()
 
    to_drop_sparse = miss[miss > DROP_SPARSE].index.tolist()

    numeric_base = [c for c in num_all if c not in to_drop_sparse]
    categorical_base = [c for c in cat_all if c not in to_drop_sparse]

    # Ensure FI_prev is present
    if "FI_prev" not in numeric_base and "FI_prev" in num_all:
        numeric_base = ["FI_prev"] + numeric_base

    X_cols_base = numeric_base + categorical_base

    # Make predictions
    X_future = df_future[X_cols_base]
    
    df_future['Predicted_FI_Rate'] = model.predict(X_future)
    # Display only selected columns

    map_html = choropleth(df_future, 'Predicted_FI_Rate', 'twilight', 'Predicted Food Insecurity Rates')

    selected_cols = ['State','County','Predicted_FI_Rate']
    df_selected = df_future[selected_cols].copy()

    def categorize(rate):
        if rate < 0.14:
            return 'Low'
        elif rate < 0.2:
            return 'Medium'
        else:
            return 'High'

    df_selected['Category'] = df_selected['Predicted_FI_Rate'].apply(categorize)
    num_high = df_selected[df_selected['Category'] == 'High'].shape[0]
        # Combine into tuples like (County, State, Level, PredictedRate, Change)
    forecasts = list(df_selected.head(20).apply(
        lambda x: (x['County'], x['State'], x['Category'], round(x['Predicted_FI_Rate'], 3)),
        axis=1
    ))
    total_counties =  len(df_future)
    table_html = df_selected.head(10).to_html(classes='data')

    return render_template(
        'results.html',
        total_counties=total_counties,
        table_html=table_html,
        map_html=map_html,
        forecasts=forecasts,
        num_high=num_high
    )



def prepareData(df):
    """
    Cleans input data, adds lag features, drops sparse columns,
    and returns (X, df_ready_for_prediction).
    """
    # IDs
    df["FIPS"] = df["FIPS"].astype(str).str.zfill(5)
    df["STATE"] = df["FIPS"].str[:2]
    df["Year"] = df["Year"].astype(int)

    # Only rows with a labeled target
    df = df[~df["FI Rate"].isna()].copy()

    # Build lag feature
    df = df.sort_values(["FIPS", "Year"])
    df["FI_prev"] = df.groupby("FIPS")["FI Rate"].shift(1)

    # Time windows
    TRAIN_START = 2010
    TEST_YEAR = 2023

    # Split only for reference — training stats are needed for feature drop
    train = df[(df["Year"] >= TRAIN_START) & (df["Year"] < TEST_YEAR)].copy()
    test  = df[df["Year"] == TEST_YEAR].copy()

    # === Identify features ===
    IGNORE = {"FIPS", "STATE", "Year", "FI Rate"}
    num_all = [c for c in df.columns if c not in IGNORE and df[c].dtype != "O"]
    cat_all = [c for c in df.columns if c not in IGNORE and df[c].dtype == "O"]

    # Drop sparse (>60% missing)
    DROP_SPARSE = 0.60
    miss = train[num_all + cat_all].isna().mean()
    to_drop_sparse = miss[miss > DROP_SPARSE].index.tolist()

    numeric_base = [c for c in num_all if c not in to_drop_sparse]
    categorical_base = [c for c in cat_all if c not in to_drop_sparse]

    # Ensure FI_prev is present
    if "FI_prev" not in numeric_base and "FI_prev" in num_all:
        numeric_base = ["FI_prev"] + numeric_base

    X_cols_base = numeric_base + categorical_base

    # Return cleaned feature matrix and aligned df
    X = test[X_cols_base].copy()
    return X, test




if __name__ == "__main__":
    app.run(debug=True)
