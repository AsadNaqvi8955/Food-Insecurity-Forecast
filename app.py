# app.py
# Food Insecurity Forecast — Community UI (Streamlit)
# Loads pickled model once; supports single & batch predictions; multipage UI.

from __future__ import annotations
import os
import io
import sys
import json
import time
import pickle
import traceback
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# Compatibility shim: some pickles reference numpy._core.* in newer numpy
# ---------------------------------------------------------------------
try:
    import numpy as _np
    import types as _types
    if not hasattr(_np, "_core"):
        _np._core = _types.SimpleNamespace()
        # minimal attributes commonly referenced
        _np._core.numeric = _np
        _np._core.fromnumeric = _np
        _np._core.arrayprint = _np
except Exception:
    pass

APP_NAME = "Food Insecurity Forecast — Community UI"
APP_VERSION = "v1.0.0"
DEFAULT_MODEL_PATH = os.environ.get(
    "FIF_MODEL_PATH",
    os.path.join("outputdata", "best_linear_model.pickle")
)
DEFAULT_SCHEMA_PATH = os.path.join("outputdata", "fully_cleaned_data_final.pickle")

TARGET_COL = "FI Rate"  # from your data
ID_LIKE = {"FIPS", "State/County", "STATE", "County"}  # will be excluded from features by default
# Columns we commonly don’t want as inputs (target, deltas, labels)
AUTO_EXCLUDE_PREFIXES = ("delta_",)
AUTO_EXCLUDE_EXACT = {TARGET_COL, "Low Threshold Type", "High Threshold Type"}

# Nice page icons
PAGE_ICONS = {
    "Home": "🏠",
    "Single Prediction": "🎯",
    "Batch Prediction": "📦",
    "Data Dictionary": "🗂️",
    "Model Card": "📄",
    "Settings & Logs": "⚙️",
    "About / Feedback": "💬",
}

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
@dataclass
class AppArtifacts:
    model: object
    model_path: str
    feature_columns: List[str]           # the columns we’ll pass into model.predict(...)
    numeric_cols: List[str]
    categorical_cols: List[str]
    schema_df_head: pd.DataFrame         # for preview & defaults
    notes: List[str]

@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> object:
    with open(model_path, "rb") as f:
        return pickle.load(f)

@st.cache_resource(show_spinner=False)
def load_schema(schema_path: str) -> pd.DataFrame:
    """
    Loads your master, fully-cleaned dataframe used during training.
    We use this only to:
      1) infer column names & types
      2) compute sensible defaults (median/mode) for UI widgets
    """
    # The file is a pickle of a pandas DataFrame
    df = pd.read_pickle(schema_path)
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Schema pickle did not contain a pandas DataFrame.")
    return df

def infer_feature_sets(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    cols = list(df.columns)
    # Drop target & known non-features
    filtered = []
    for c in cols:
        if c in AUTO_EXCLUDE_EXACT: 
            continue
        if any(c.startswith(p) for p in AUTO_EXCLUDE_PREFIXES):
            continue
        filtered.append(c)
    # Put IDs last in case user wants to keep but we default to excluding them for model inputs
    ordered = [c for c in filtered if c not in ID_LIKE] + [c for c in filtered if c in ID_LIKE]

    numeric_cols = [c for c in ordered if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in ordered if not pd.api.types.is_numeric_dtype(df[c])]
    # Default model feature columns (exclude id-like by default)
    model_features = [c for c in ordered if c not in ID_LIKE]
    return model_features, numeric_cols, categorical_cols

def df_defaults(df: pd.DataFrame) -> Dict[str, object]:
    defaults = {}
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            defaults[c] = float(df[c].median(skipna=True))
        else:
            # use most common category/string, fallback to first non-null
            mode = df[c].mode(dropna=True)
            if len(mode):
                defaults[c] = str(mode.iloc[0])
            else:
                first = df[c].dropna().astype(str)
                defaults[c] = str(first.iloc[0]) if len(first) else ""
    return defaults

def align_for_model(input_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    # Keep only model features; add any missing columns as NaN (model’s pipeline should impute/handle)
    out = input_df.copy()
    # Drop target if user uploaded it
    for t in (TARGET_COL,):
        if t in out.columns:
            out = out.drop(columns=[t])
    # Keep model columns in order
    for c in feature_cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out[feature_cols]
    return out

def predict_df(model, X: pd.DataFrame) -> np.ndarray:
    # Try predict_proba if classification; else predict (this is a regressor)
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] >= 1:
                return proba[:, -1]
        except Exception:
            pass
    return model.predict(X)

# ---------------------------------------------------------------------
# Load artifacts once
# ---------------------------------------------------------------------
def init_artifacts(model_path: str, schema_path: str) -> AppArtifacts:
    notes = []
    model = load_model(model_path)
    schema_df = load_schema(schema_path)

    features, numeric_cols, categorical_cols = infer_feature_sets(schema_df)
    if not len(features):
        raise ValueError("No usable feature columns inferred from schema.")

    # preview first N rows for defaults / examples
    head = schema_df.head(50).copy()

    notes.append(f"Loaded model from: {model_path}")
    notes.append(f"Loaded schema from: {schema_path}")
    notes.append(f"Inferred {len(features)} model feature columns "
                 f"({len(numeric_cols)} numeric, {len(categorical_cols)} categorical)")
    return AppArtifacts(
        model=model,
        model_path=model_path,
        feature_columns=features,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        schema_df_head=head,
        notes=notes
    )

# ---------------------------------------------------------------------
# Page: Home
# ---------------------------------------------------------------------
def page_home(art: AppArtifacts):
    st.markdown(f"# {PAGE_ICONS['Home']} Welcome")
    st.write(
        """
        This app lets community users explore **Food Insecurity Rate** predictions using a
        pre-trained model. It **does not retrain**; it loads your saved model artifact and runs
        inference on new data.
        """
    )
    with st.expander("What can I do here?"):
        st.markdown(
            """
            - **🎯 Single Prediction:** Fill in a form and get a prediction instantly  
            - **📦 Batch Prediction:** Upload a CSV and download predictions  
            - **🗂️ Data Dictionary:** See feature list and types inferred from your training data  
            - **📄 Model Card:** Assumptions, limitations, and version info  
            - **⚙️ Settings & Logs:** Change paths and review load notes  
            - **💬 About / Feedback:** Share ideas to improve the community tool
            """
        )

    st.subheader("Quick preview of schema (sample rows)")
    st.dataframe(art.schema_df_head, use_container_width=True)
    st.caption("Preview comes from `fully_cleaned_data_final.pickle` (first ~50 rows).")

# ---------------------------------------------------------------------
# Page: Single Prediction (interactive form)
# ---------------------------------------------------------------------
def page_single(art: AppArtifacts):
    st.markdown(f"## {PAGE_ICONS['Single Prediction']} Single Prediction")

    # Build defaults from sample schema
    defaults = df_defaults(art.schema_df_head)
    # Only present model features as inputs
    form_cols = art.feature_columns

    st.info("Fill in the fields below. Leave blanks if unsure (the model pipeline will impute).")
    # Two-column responsive layout
    cols = st.columns(2)
    user_inputs: Dict[str, object] = {}
    for i, col_name in enumerate(form_cols):
        col = cols[i % 2]
        with col:
            if col_name in art.numeric_cols:
                val = st.number_input(
                    label=col_name,
                    value=float(defaults.get(col_name, 0.0)),
                    step=0.1,
                    format="%.6f"
                )
            else:
                # Categorical: allow free text (since real set may be large)
                val = st.text_input(
                    label=col_name,
                    value=str(defaults.get(col_name, "")),
                    placeholder="e.g., CA / Los Angeles / Urban …"
                )
            user_inputs[col_name] = val

    # Predict
    if st.button("Predict FI Rate", type="primary"):
        try:
            X = pd.DataFrame([user_inputs])
            X = align_for_model(X, art.feature_columns)
            y_pred = predict_df(art.model, X)
            pred = float(y_pred[0])
            st.success(f"Predicted **Food Insecurity Rate**: **{pred:.4f}**")
            st.caption("Note: Model returns a continuous rate; display is clipped to 4 decimals.")

            # Show the exact row fed to model (post-alignment)
            with st.expander("See model-ready row"):
                st.dataframe(X.T, use_container_width=True)
        except Exception as e:
            st.error("Prediction failed. See details in the panel below.")
            with st.expander("Error details"):
                st.code("".join(traceback.format_exception(e)), language="python")

# ---------------------------------------------------------------------
# Page: Batch Prediction (CSV)
# ---------------------------------------------------------------------
def page_batch(art: AppArtifacts):
    st.markdown(f"## {PAGE_ICONS['Batch Prediction']} Batch Prediction")
    st.write("Upload a CSV with columns matching the model’s expected feature names.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    keep_id_cols = st.toggle("Include ID-like columns (e.g., FIPS) in the output", value=True)

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.success(f"Loaded file with shape {df.shape}")
            st.dataframe(df.head(), use_container_width=True)
            # Align and predict
            X = align_for_model(df, art.feature_columns)
            y_pred = predict_df(art.model, X)
            out = df.copy()
            out["Predicted FI Rate"] = y_pred

            # Optionally keep only IDs + prediction
            if keep_id_cols:
                id_cols = [c for c in df.columns if c in ID_LIKE]
                if id_cols:
                    out = out[id_cols + [c for c in df.columns if c not in id_cols and c != TARGET_COL] + ["Predicted FI Rate"]]

            # Download
            buff = io.StringIO()
            out.to_csv(buff, index=False)
            st.download_button(
                "Download predictions CSV",
                buff.getvalue(),
                file_name="fi_predictions.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error("Batch prediction failed.")
            with st.expander("Error details"):
                st.code("".join(traceback.format_exception(e)), language="python")

    with st.expander("Expected feature list"):
        st.code(json.dumps(art.feature_columns, indent=2))

# ---------------------------------------------------------------------
# Page: Data Dictionary
# ---------------------------------------------------------------------
def page_dictionary(art: AppArtifacts):
    st.markdown(f"## {PAGE_ICONS['Data Dictionary']} Data Dictionary")
    st.write("Feature types are inferred from your training dataframe.")
    dd = []
    for c in art.feature_columns:
        dtype = "numeric" if c in art.numeric_cols else "categorical"
        dd.append((c, dtype))
    dict_df = pd.DataFrame(dd, columns=["feature", "type"])
    st.dataframe(dict_df, use_container_width=True)
    st.caption("Source: `fully_cleaned_data_final.pickle` (head sample), filtered to exclude targets/labels.")

# ---------------------------------------------------------------------
# Page: Model Card
# ---------------------------------------------------------------------
def page_model_card(art: AppArtifacts):
    st.markdown(f"## {PAGE_ICONS['Model Card']} Model Card")
    st.write(f"**Artifact:** `{art.model_path}`")
    st.write("**App version:**", APP_VERSION)

    with st.expander("Intended use"):
        st.markdown(
            """
            - Forecast **Food Insecurity Rate** for counties/regions, given socio-economic and cost features.  
            - Supports both **single** and **batch** (CSV) inference.  
            - Not for clinical or life-critical use.
            """
        )
    with st.expander("Inputs & schema"):
        st.write(f"Number of features passed to the model: `{len(art.feature_columns)}`")
        st.code(json.dumps(art.feature_columns[:50], indent=2) + ("\n...\n" if len(art.feature_columns) > 50 else ""))

    with st.expander("Assumptions & caveats"):
        st.markdown(
            """
            - Assumes the **same feature engineering** and column definitions used in training.  
            - Pipelines inside the pickle should handle imputation/encoding.  
            - Predictions outside the training distribution may be unreliable.  
            - Version mismatches between **numpy/scikit-learn** can break unpickling. Align env if needed.
            """
        )

# ---------------------------------------------------------------------
# Page: Settings & Logs
# ---------------------------------------------------------------------
def page_settings(art_state: Dict[str, str], notes: List[str]):
    st.markdown(f"## {PAGE_ICONS['Settings & Logs']} Settings & Logs")
    st.write("Change paths and reload artifacts if needed.")

    with st.form("paths"):
        model_path = st.text_input("Model pickle path", art_state.get("model_path", DEFAULT_MODEL_PATH))
        schema_path = st.text_input("Schema pickle path (DataFrame)", art_state.get("schema_path", DEFAULT_SCHEMA_PATH))
        submitted = st.form_submit_button("Save & Reload", type="primary")
    if submitted:
        st.session_state["_model_path"] = model_path.strip()
        st.session_state["_schema_path"] = schema_path.strip()
        st.success("Saved. Use the button below to reload.")
    if st.button("Reload artifacts"):
        st.cache_resource.clear()
        st.experimental_rerun()

    st.subheader("Load Notes")
    if notes:
        st.code("\n".join(notes))
    else:
        st.write("No notes recorded.")

# ---------------------------------------------------------------------
# Page: About / Feedback
# ---------------------------------------------------------------------
def page_about():
    st.markdown(f"## {PAGE_ICONS['About / Feedback']} About / Feedback")
    st.write(
        """
        This UI is designed for **community use and extension**.  
        Ideas to extend next:
        - Lightweight **Explainability**: global feature importance (model-agnostic permutation)  
        - **Scenario builder**: adjust key drivers and compare deltas  
        - **Role-based views** for councils/NGOs vs. analysts  
        """
    )
    st.write("Share feedback below (stored only in your session):")
    fb = st.text_area("Your thoughts or feature requests")
    if st.button("Save note to session"):
        st.session_state.setdefault("feedback_notes", []).append({"ts": time.time(), "note": fb})
        st.success("Saved to this session.")
    with st.expander("Session notes"):
        st.json(st.session_state.get("feedback_notes", []))

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="🍽️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.sidebar.title(APP_NAME)
    st.sidebar.caption(APP_VERSION)

    # Remember paths between reruns
    model_path = st.session_state.get("_model_path", DEFAULT_MODEL_PATH)
    schema_path = st.session_state.get("_schema_path", DEFAULT_SCHEMA_PATH)

    # Load artifacts (one time via cache)
    try:
        art = init_artifacts(model_path, schema_path)
        # Expose simple state for Settings page
        art_state = {"model_path": model_path, "schema_path": schema_path}
    except Exception as e:
        st.error("Failed to load model/schema. Fix paths or environment and reload.")
        with st.expander("Error details"):
            st.code("".join(traceback.format_exception(e)), language="python")
        st.stop()

    # Sidebar nav
    page = st.sidebar.radio(
        "Navigate",
        list(PAGE_ICONS.keys()),
        format_func=lambda x: f"{PAGE_ICONS[x]} {x}"
    )

    # Route
    if page == "Home":
        page_home(art)
    elif page == "Single Prediction":
        page_single(art)
    elif page == "Batch Prediction":
        page_batch(art)
    elif page == "Data Dictionary":
        page_dictionary(art)
    elif page == "Model Card":
        page_model_card(art)
    elif page == "Settings & Logs":
        page_settings({"model_path": model_path, "schema_path": schema_path}, art.notes)
    elif page == "About / Feedback":
        page_about()
    else:
        st.write("Unknown page.")

if __name__ == "__main__":
    main()
