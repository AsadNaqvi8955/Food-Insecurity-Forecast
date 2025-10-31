Food Insecurity Forecasting for U.S. Counties

This project predicts county-level food insecurity rates across the United States to support nonprofits, food banks, and community kitchens in understanding and planning for food demand.
The predictive model leverages socioeconomic data (e.g., rent, unemployment, cost per meal) and provides an interactive Streamlit dashboard for scenario forecasting and visualization.

1. Project Overview

Food insecurity affects millions of households across the U.S., and forecasting these patterns can help local organizations allocate resources more effectively.
This project aims to build a data-driven model that forecasts future food insecurity rates at the county level and presents the insights through an interactive web application.

2. Data Pipeline
Data Collection

Multiple datasets were gathered from government and public data portals, including:

U.S. Department of Agriculture (USDA)

U.S. Census Bureau

U.S. Bureau of Labor Statistics

Department of Housing and Urban Development (HUD)

Data Integration and Cleaning

Datasets were unified using FIPS codes (Federal Information Processing Standards) to ensure consistent county-level mapping.

Missing values and inconsistencies were addressed.

Derived new variables such as lag features (FI_prev) to capture temporal patterns.

Feature Engineering

Created engineered metrics for rent, unemployment, and cost per meal trends.

Applied normalization and transformation where necessary.

3. Model Development

Several regression models were tested and compared using R² and RMSE metrics on both training and test data.
The results are summarized below:

Model	R² (Train)	RMSE (Train)	R² (Test)	RMSE (Test)	CV R² Mean	CV R² Std	CV RMSE Mean	CV RMSE Std
M1: OLS (All Features + FI_prev)	0.989	0.0044	0.995	0.0025	0.974	0.014	0.0060	0.0013
M2: OLS (VIF ≤ 10 + FI_prev, n_num=32)	0.989	0.0044	0.995	0.0026	0.976	0.012	0.0058	0.0012
M3: XGBoost (+FI_prev)	0.993	0.0035	0.993	0.0029	0.982	0.004	0.0054	0.0012
M4: CatBoost (+FI_prev)	0.991	0.0040	0.985	0.0043	0.984	0.004	0.0049	0.0009
M5: Random Forest (+FI_prev)	0.995	0.0029	0.993	0.0029	—	—	—	—
M6: Hybrid (OLS + XGB + CatB + StateCorr + FI_prev)	—	—	0.994	0.0028	—	—	—	—

Selected Model:
Model 1 (OLS with all features including FI_prev) achieved high performance and was chosen for deployment due to its interpretability and stability across states.

4. Streamlit Application

The model was integrated into a Streamlit app for interactive exploration and scenario simulation.

Launch Instructions

Run the Streamlit app:

streamlit run app.py


5. Outcomes

Developed an accurate and explainable forecasting model for food insecurity.

Built an interactive, user-friendly dashboard for NGOs and food banks.

Provided a scalable foundation for future integration with live socioeconomic data.

6. Future Improvements

Integrate live economic indicators (e.g., from APIs).

Include time-series models for multi-year trend forecasting.
