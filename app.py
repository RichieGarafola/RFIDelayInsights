# 🔧 Imports
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

# -------------------------------------
# Page Configuration and Global Styling
# -------------------------------------
# Configure the Streamlit page with a custom title and wide layout
st.set_page_config(page_title="RFI Delay Insights", layout="wide")

# Inject custom CSS to adjust font and layout padding for better aesthetics
st.markdown("""
<style>
    /* Set default font */
    body { font-family: 'Segoe UI', sans-serif; }
    
    /* Add vertical padding around main content */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    
    /* Tighten spacing below headers */
    h1, h2, h3 { margin-bottom: 0.3em; }                            
</style>
""", unsafe_allow_html=True)


# -------------------------------------
# Cached Data Load and Preprocessing
# -------------------------------------
# Cache the data loading and transformation to improve performance on repeated runs
@st.cache_data
def load_and_process_data(path: str) -> pd.DataFrame:
    # Load CSV and parse dates for time-based analysis
    df = pd.read_csv(path, parse_dates=['submission_date', 'response_date'])
    
    # Calculate number of days taken to close each RFI
    df['days_to_close'] = (df['response_date'] - df['submission_date']).dt.days
    
    # Extract submission month as a string for time-series grouping
    df['month'] = df['submission_date'].dt.to_period('M').astype(str)
    
    # Normalize delay_flag: fill missing with 0, ensure binary, and clamp values
    df['delay_flag'] = df['delay_flag'].fillna(0).astype(int).clip(0, 1)
    
    # Create a human-readable label for delay status
    df['delay_flag_label'] = df['delay_flag'].map({0: 'No Delay', 1: 'Delay'})
    
    return df


# -------------------------------------
# Logistic Regression Model Trainer
# -------------------------------------
def train_logistic_model(df: pd.DataFrame):
    # One-hot encode categorical features, dropping the first level to avoid multicollinearity
    df_model = pd.get_dummies(df, columns=[
        'status', 'topic', 'assigned_to', 'priority_level',
        'project_phase', 'rfi_type'
    ], drop_first=True)

    # Define target variable
    y = df_model['delay_flag']

    # Drop non-predictive or leakage-prone columns
    drop_cols = ['delay_flag', 'rfi_id', 'project_name', 'description',
                 'submission_date', 'response_date', 'delay_flag_label', 'month']
    X = df_model.drop(columns=[col for col in drop_cols if col in df_model.columns])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Standardize features to improve model performance and convergence
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Predict outcomes and probabilities on the test set
    y_pred = model.predict(X_test_scaled)
    y_probs = model.predict_proba(X_test_scaled)[:, 1]  # Probabilities for class '1' (Delay)

    # Return model, feature names, and key performance metrics
    return model, X.columns, accuracy_score(y_test, y_pred), roc_auc_score(y_test, y_probs), y_probs

# -------------------------------------
# Executive Summary Generator
# -------------------------------------
def generate_rfi_summary(df):
    if df.empty:
        return "<b>No data available to generate summary.</b>"

    try:
        # Total number of RFIs
        total_rfis = len(df)

        # Handle missing submission dates
        if df['submission_date'].isna().all():
            date_min = "N/A"
            date_max = "N/A"
        else:
            date_min = df['submission_date'].min().strftime('%B %Y')
            date_max = df['submission_date'].max().strftime('%B %Y')

        # Average time to close
        avg_days_close = round(df['days_to_close'].mean(), 1) if not df['days_to_close'].isna().all() else "N/A"

        # Delay rate
        delay_rate = round(df['delay_flag'].mean() * 100, 1) if 'delay_flag' in df else "N/A"

        # Top 3 projects
        top_projects = df['project_name'].value_counts().head(3).index.tolist()
        top_projects += ["N/A"] * (3 - len(top_projects))

        # Top 3 topics
        top_topics = df['topic'].value_counts().head(3).index.tolist()
        top_topics += ["N/A"] * (3 - len(top_topics))

        # Cost by priority
        cost_by_priority = df.groupby('priority_level')['cost_impact_estimate'].median().to_dict()
        cost_high = f"${cost_by_priority.get('High', 0):,.0f}"
        cost_med = f"${cost_by_priority.get('Medium', 0):,.0f}"
        cost_low = f"${cost_by_priority.get('Low', 0):,.0f}"

        # Delay likelihood by priority
        delay_by_priority = df.groupby('priority_level')['delay_flag'].mean()
        if delay_by_priority.get('Low', 0):
            delay_ratio = round(delay_by_priority.get('High', 0) / delay_by_priority.get('Low', 0), 1)
        else:
            delay_ratio = "N/A"

        # Top 2 phases with highest delay rates
        delay_by_phase = df.groupby('project_phase')['delay_flag'].mean().sort_values(ascending=False)
        top_phases = delay_by_phase.head(2).index.tolist()
        top_phases += ["N/A"] * (2 - len(top_phases))

        # HTML Summary
        return f"""
<b>Between <u>{date_min}</u> and <u>{date_max}</u></b>, a total of <b>{total_rfis:,}</b> RFIs were submitted across major projects, including:
<ul>
  <li><i>{top_projects[0]}</i></li>
  <li><i>{top_projects[1]}</i></li>
  <li><i>{top_projects[2]}</i></li>
</ul>
<b>Key Insights:</b>
<ul>
  <li><b>Top Topics:</b> {top_topics[0]}, {top_topics[1]}, {top_topics[2]}</li>
  <li><b>Avg. Time to Close:</b> {avg_days_close} days</li>
  <li><b>Delay Rate:</b> {delay_rate}%</li>
  <li><b>Cost Impact:</b> High: {cost_high} | Medium: {cost_med} | Low: {cost_low}</li>
  <li><b>Delay Risk:</b> High-priority RFIs are <b>{delay_ratio}×</b> more likely to delay than Low</li>
  <li><b>High-Risk Phases:</b> {top_phases[0]}, {top_phases[1]}</li>
</ul>
"""
    except Exception as e:
        return f"<b>Summary generation failed:</b> {str(e)}"


# ======================
# 📊 Dashboard Interface
# ======================

##############
# Load Data
##############
df = load_and_process_data("roadside_rfi.csv")

##############
# Sidebar Title and Filters
##############
# Sidebar title for the RFI dashboard
st.sidebar.title("🏗️ RFI Insights Dashboard")

# Extract unique values for topic and priority to populate filter options
topics = df["topic"].unique().tolist()
priorities = df["priority_level"].unique().tolist()

# Initialize filter session state if not already set (for persistence)
if "topic_filter" not in st.session_state:
    st.session_state.topic_filter = topics
if "priority_filter" not in st.session_state:
    st.session_state.priority_filter = priorities

# Filter header and multiselect UI elements
st.sidebar.header("🔎 Filters")
st.sidebar.multiselect("Topic", options=topics, default=st.session_state.topic_filter, key="topic_filter")
st.sidebar.multiselect("Priority", options=priorities, default=st.session_state.priority_filter, key="priority_filter")

# Reset function to restore filters to full dataset state
def reset_filters():
    st.session_state.topic_filter = topics
    st.session_state.priority_filter = priorities

# Button to trigger filter reset
st.sidebar.button("Reset Filters", on_click=reset_filters)

# Sidebar expander to display a lightweight data quality audit
with st.sidebar.expander("🧪 Data Audit", expanded=False):
    # Basic record and missing data counts
    total_records = len(df)
    missing_flags = df['delay_flag'].isna().sum()
    non_binary_flags = df[~df['delay_flag'].isin([0, 1])].shape[0]
    missing_dates = df[['submission_date', 'response_date']].isna().sum()

    # Render audit stats
    st.markdown(f"**Total Records:** {total_records:,}")
    st.markdown(f"- Missing `delay_flag`: **{missing_flags:,}**")
    st.markdown(f"- Non-binary `delay_flag`: **{non_binary_flags:,}**")
    st.markdown(f"- Missing `submission_date`: **{missing_dates['submission_date']}**")
    st.markdown(f"- Missing `response_date`: **{missing_dates['response_date']}**")

    # Scoring system: deduct based on data quality issues
    score = 100
    penalties = 0
    if missing_flags > 0: penalties += 10
    if non_binary_flags > 0: penalties += 15
    if missing_dates['submission_date'] > 0 or missing_dates['response_date'] > 0: penalties += 15
    score = max(0, score - penalties)

    # Determine rating category based on score
    rating = "✅ Good" if score >= 90 else "⚠️ Moderate" if score >= 70 else "❌ Poor"

    # Display audit rating and progress bar
    st.markdown("---")
    st.markdown(f"**Audit Rating:** {rating}")
    st.progress(score)

##############
# Filtered Data
##############
# Apply topic and priority filters selected in the sidebar to the main dataset
df_filtered = df[
    (df['topic'].isin(st.session_state.topic_filter)) &
    (df['priority_level'].isin(st.session_state.priority_filter))
]

##############
# KPI Metrics
##############
# If no records match the selected filters, show a warning and halt execution
if df_filtered.empty:
    st.warning("⚠️ No RFIs match the current filter selection. Please adjust the filters to view insights.")
    st.button("🔁 Reset Filters", on_click=reset_filters)
    st.stop()  # Prevent further rendering when there's no data

else:
    # Display key performance indicators (KPIs) in an expandable section
    with st.expander("📊 KPI Metrics", expanded=True):
        col1, col2, col3, col4 = st.columns(4)  # Layout KPIs in 4 equal columns

        # Show total number of filtered RFIs
        col1.metric("📌 Total RFIs", f"{len(df_filtered):,}")

        # Show average number of days it takes to close an RFI
        col2.metric("⏱ Avg Days to Close", f"{round(df_filtered['days_to_close'].mean(), 1)} days")

        # Show percentage of RFIs marked as delayed
        col3.metric("⚠️ Delay Rate", f"{round(df_filtered['delay_flag'].mean() * 100, 1)}%")

        # Show median cost impact of filtered RFIs (handle nulls gracefully)
        median_cost = df_filtered['cost_impact_estimate'].median()
        col4.metric("💰 Median Cost Impact", f"${int(median_cost):,}" if pd.notnull(median_cost) else "N/A")

##############
# Build Visuals
##############
# 📊 Box plot: Cost impact by priority level, colored by delay status
fig1 = px.box(df_filtered, x='priority_level', y='cost_impact_estimate', color='delay_flag_label')
fig1.update_layout(
    title_text="💰 Cost Impact by Priority & Delay",
    title_x=0.02,
    margin=dict(t=40),
    legend_title_text="Delay"
)

# 📊 Histogram: Frequency of delays grouped by topic
fig2 = px.histogram(df_filtered, x='topic', color='delay_flag_label', barmode='group')
fig2.update_layout(
    title_text="📊 Delay Frequency by Topic",
    title_x=0.02,
    xaxis_tickangle=-45,
    legend_title_text="Delay"
)

# 📈 Line chart: Trend of average monthly delay rate
fig3 = px.line(
    df_filtered.groupby('month')['delay_flag'].mean().reset_index(),
    x='month', y='delay_flag'
)
fig3.update_layout(title_text="📈 Monthly Delay Rate Trend", title_x=0.02)

# 🤖 Train logistic regression model on filtered dataset and capture evaluation metrics
model, features, acc, auc, y_probs = train_logistic_model(df_filtered)

# 📊 Histogram: Predicted probabilities for delay from the logistic model
# Convert predicted probabilities to a DataFrame for clearer labeling
y_prob_df = pd.DataFrame({'Predicted Probability': y_probs})

# Plot histogram with explicit axis label
fig4 = px.histogram(y_prob_df, x='Predicted Probability', nbins=30)
fig4.update_layout(
    title_text="🤖 Predicted Delay Probability Distribution",
    title_x=0.02,
    xaxis_title="Probability of Delay",
    yaxis_title="Count",
    showlegend=False
)

##############
# Visual Quadrants
##############
# Expandable section for visualizing both exploratory and predictive charts
with st.expander("📈 Exploratory & Predictive Charts", expanded=True):
    
    # Top row: Boxplot (Cost vs. Priority) and Histogram (Delays by Topic)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)  # Cost impact by priority and delay
    with col2:
        st.plotly_chart(fig2, use_container_width=True)  # Delay frequency by topic

    # Bottom row: Delay rate trend and prediction probability histogram
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(fig3, use_container_width=True)  # Monthly delay rate trend
    with col4:
        with st.container():
            # Show model performance metrics directly below the chart
            kpi_col = st.columns([1, 1])
            kpi_col[0].metric("Accuracy", f"{round(acc*100, 2)}%")
            kpi_col[1].metric("ROC AUC", f"{round(auc, 3)}")
            
            # Display histogram of predicted delay probabilities
            st.plotly_chart(fig4, use_container_width=True)
            
##############
# Executive Summary
##############
# Expandable section to display a narrative executive summary of the filtered RFI data
with st.expander("📄 Executive Summary", expanded=False):
    
    # Generate the summary text (HTML formatted) from filtered data
    summary = generate_rfi_summary(df_filtered)

    # Render the summary with support for HTML formatting
    st.markdown(summary, unsafe_allow_html=True)

    # Provide an option to download the summary as a text file
    st.download_button(
        "📥 Download Summary",
        summary,
        file_name="rfi_summary.txt",
        key="download_summary_button"
    )

##############
# Filtered Table
##############
# Expandable section to preview a subset of the filtered RFI records in tabular form
with st.expander("📋 Filtered RFI Snapshot", expanded=False):
    
    # Display selected columns from the filtered dataset for quick review
    st.dataframe(
        df_filtered[['rfi_id', 'project_name', 'topic', 'priority_level', 'cost_impact_estimate', 'delay_flag_label']],
        
        # Ensures table stretches to container width for better readability
        use_container_width=True 
    )
