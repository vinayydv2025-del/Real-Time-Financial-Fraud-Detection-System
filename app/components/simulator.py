
from pathlib import Path

import pandas as pd
import streamlit as st

from app.api_client import (
    check_api_health,
    predict_transaction,
)


BASE_DIR = Path(__file__).resolve().parents[2]

TEST_DATA_PATH = BASE_DIR / "data" / "simulation" / "simulation_transactions.csv"


# ============================================================
# LOAD HISTORICAL TRANSACTIONS
# ============================================================

@st.cache_data(show_spinner="Loading historical transactions...")
def load_simulation_data():

    data = pd.read_csv(TEST_DATA_PATH)

    required_columns = [
        "Time",
        "Amount",
        "Class",
        *[f"V{i}" for i in range(1, 29)],
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing dataset columns: {missing_columns}"
        )

    # Replay transactions chronologically.
    data = data.sort_values(
        "Time",
        kind="stable"
    ).reset_index(drop=True)

    return data


# ============================================================
# PREPARE API REQUEST
# ============================================================

def prepare_api_transaction(row):

    transaction = {
        "Time": float(row["Time"]),
        "Amount": float(row["Amount"]),
    }

    for i in range(1, 29):

        transaction[f"V{i}"] = float(
            row[f"V{i}"]
        )

    # IMPORTANT:
    # Class is deliberately excluded.
    # The model must predict the result itself.

    return transaction


# ============================================================
# CALCULATE SIMULATION PERFORMANCE
# ============================================================

def calculate_simulation_metrics(results):

    predictions = pd.DataFrame(results)

    if predictions.empty:
        return {}

    actual = predictions["actual_class"].astype(int)
    predicted = predictions["prediction"].astype(int)

    tp = int(
        ((actual == 1) & (predicted == 1)).sum()
    )

    fp = int(
        ((actual == 0) & (predicted == 1)).sum()
    )

    fn = int(
        ((actual == 1) & (predicted == 0)).sum()
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0
    )

    return {
        "transactions": len(predictions),
        "fraud_alerts": int(predicted.sum()),
        "actual_frauds": int(actual.sum()),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
    }


# ============================================================
# MAIN SIMULATOR
# ============================================================

def render_simulator():

    st.subheader("Real-Time Transaction Simulator")

    st.caption(
        "Replay historical credit card transactions through "
        "the FastAPI fraud detection model."
    )

    st.info(
        "Simulation mode: transactions are sent sequentially "
        "to the prediction API. Actual fraud labels are used "
        "only for post-prediction comparison."
    )

    # --------------------------------------------------------
    # CHECK API CONNECTION
    # --------------------------------------------------------

    api_status = check_api_health()

    if api_status["online"]:

        st.success(
            f"Prediction API online | Model: {api_status['model']}"
        )

    else:

        st.error(
            f"Prediction API offline: {api_status['message']}"
        )

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    try:

        data = load_simulation_data()

    except Exception as error:

        st.error(
            f"Could not load simulation dataset: {error}"
        )
        return

    total_available = len(data)

    # --------------------------------------------------------
    # INITIALIZE SESSION STATE
    # --------------------------------------------------------

    if "simulation_cursor" not in st.session_state:

        st.session_state.simulation_cursor = 0

    if "simulation_results" not in st.session_state:

        st.session_state.simulation_results = []

    cursor = st.session_state.simulation_cursor

    # --------------------------------------------------------
    # SIMULATION STATUS
    # --------------------------------------------------------

    st.divider()

    st.markdown("### Simulation Status")

    status_col1, status_col2, status_col3 = st.columns(3)

    status_col1.metric(
        "Available Transactions",
        f"{total_available:,}",
    )

    status_col2.metric(
        "Transactions Processed",
        f"{cursor:,}",
    )

    status_col3.metric(
        "Transactions Remaining",
        f"{max(total_available - cursor, 0):,}",
    )

    # --------------------------------------------------------
    # BATCH SIZE
    # --------------------------------------------------------

    st.markdown("### Configure Transaction Replay")

    batch_size = st.selectbox(
        "Transactions per batch",
        options=[5, 10, 25, 50, 100],
        index=1,
    )

    st.caption(
        "Transactions are processed in chronological order. "
        "Keep the FastAPI server running throughout the replay."
    )

    # --------------------------------------------------------
    # RUN SIMULATION
    # --------------------------------------------------------

    simulation_complete = cursor >= total_available

    if st.button(
        "Run Next Transaction Batch",
        type="primary",
        disabled=(
            not api_status["online"]
            or simulation_complete
        ),
        use_container_width=True,
    ):

        start_index = st.session_state.simulation_cursor

        end_index = min(
            start_index + batch_size,
            total_available,
        )

        batch = data.iloc[
            start_index:end_index
        ]

        progress = st.progress(
            0,
            text="Starting transaction replay..."
        )

        new_results = []

        for position, (_, row) in enumerate(
            batch.iterrows(),
            start=1,
        ):

            try:

                # Send only raw transaction features.
                payload = prepare_api_transaction(row)

                prediction = predict_transaction(
                    payload
                )

                # Read the actual label only AFTER prediction.
                actual_class = int(row["Class"])

                result_record = {
                    "transaction_id": prediction["transaction_id"],
                    "Time": float(row["Time"]),
                    "amount": float(row["Amount"]),
                    "actual_class": actual_class,
                    "prediction": int(prediction["prediction"]),
                    "result": prediction["result"],
                    "fraud_probability": float(
                        prediction["fraud_probability"]
                    ),
                    "risk_level": prediction["risk_level"],
                    "response_time_ms": float(
                        prediction["response_time_ms"]
                    ),
                }

                new_results.append(result_record)

                progress.progress(
                    position / len(batch),
                    text=(
                        f"Processed {position} "
                        f"of {len(batch)} transactions"
                    ),
                )

            except Exception as error:

                st.error(
                    f"Simulation stopped at dataset row "
                    f"{start_index + position}: {error}"
                )

                break

        # Preserve successful predictions.
        st.session_state.simulation_results.extend(
            new_results
        )

        st.session_state.simulation_cursor += len(
            new_results
        )

        if new_results:

            st.success(
                f"Successfully processed {len(new_results)} "
                "transactions. Predictions were saved through the API."
            )

        if len(new_results) < len(batch):

            st.warning(
                "The remaining transactions in this batch "
                "were not processed. Resolve the API error "
                "before continuing."
            )

        st.rerun()

    if simulation_complete:

        st.success(
            "All available test transactions have been replayed."
        )

    # --------------------------------------------------------
    # SIMULATION RESULTS
    # --------------------------------------------------------

    results = st.session_state.simulation_results

    if not results:
        st.info(
            "No simulation results yet. Start the API and run "
            "your first transaction batch."
        )
        return

    st.divider()

    st.markdown("### Simulation Performance")

    metrics = calculate_simulation_metrics(results)

    metric_col1, metric_col2, metric_col3, metric_col4 = (
        st.columns(4)
    )

    metric_col1.metric(
        "Processed",
        f"{metrics['transactions']:,}",
    )

    metric_col2.metric(
        "Fraud Alerts",
        f"{metrics['fraud_alerts']:,}",
    )

    metric_col3.metric(
        "Actual Fraud Cases",
        f"{metrics['actual_frauds']:,}",
    )

    metric_col4.metric(
        "True Detections",
        f"{metrics['true_positives']:,}",
    )

    performance_col1, performance_col2 = st.columns(2)

    performance_col1.metric(
        "Simulation Precision",
        f"{metrics['precision']:.2%}",
    )

    performance_col2.metric(
        "Simulation Recall",
        f"{metrics['recall']:.2%}",
    )

    st.caption(
        "These metrics describe only the transactions replayed "
        "in this simulation session. They are not replacements "
        "for the held-out test evaluation."
    )

    # --------------------------------------------------------
    # RECENT PREDICTIONS
    # --------------------------------------------------------

    st.markdown("### Recent Simulation Predictions")

    results_df = pd.DataFrame(results)

    st.dataframe(
        results_df.tail(25).iloc[::-1],
        use_container_width=True,
        hide_index=True,
        column_config={
            "transaction_id": "Database Transaction ID",
            "Time": "Elapsed Time",
            "amount": st.column_config.NumberColumn(
                "Amount",
                format="%.2f",
            ),
            "actual_class": "Actual Class",
            "prediction": "Predicted Class",
            "result": "Model Result",
            "fraud_probability": st.column_config.ProgressColumn(
                "Fraud Probability",
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
            ),
            "risk_level": "Risk Level",
            "response_time_ms": st.column_config.NumberColumn(
                "API Latency (ms)",
                format="%.2f",
            ),
        },
    )

    # --------------------------------------------------------
    # CONFUSION COUNTS
    # --------------------------------------------------------

    st.markdown("### Simulation Detection Outcomes")

    outcome_col1, outcome_col2, outcome_col3 = st.columns(3)

    outcome_col1.metric(
        "True Positives",
        metrics["true_positives"],
    )

    outcome_col2.metric(
        "False Positives",
        metrics["false_positives"],
    )

    outcome_col3.metric(
        "False Negatives",
        metrics["false_negatives"],
    )
