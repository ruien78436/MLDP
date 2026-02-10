import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(
    page_title="Apex Strategy AI",
    page_icon="🏎️",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    model = joblib.load('f1_position_model.pkl')
    model_columns = joblib.load('f1_model_columns.pkl')
    return model, model_columns

try:
    model, model_columns = load_artifacts()
except FileNotFoundError:
    st.error("Error: Model files not found. Please run your Jupyter Notebook to generate 'f1_position_model.pkl' and 'f1_model_columns.pkl'.")
    st.stop()

st.sidebar.header("RACE STRATEGY CONFIG")
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg", width=100)

teams = ['Red Bull', 'Ferrari', 'Mercedes', 'McLaren', 'Aston Martin', 'Alpine', 'Williams', 'Haas F1 Team', 'Kick Sauber', 'RB']
circuits = ['Circuit de Monaco', 'Silverstone Circuit', 'Monza', 'Spa-Francorchamps', 'Marina Bay Street Circuit', 'Suzuka Circuit']

selected_team = st.sidebar.selectbox("Select Constructor", teams)
selected_circuit = st.sidebar.selectbox("Select Circuit", circuits)
grid_position = st.sidebar.slider("Starting Grid Position", 1, 20, 1)
qualifying_pos = st.sidebar.slider("Qualifying Position", 1, 20, grid_position) # Default to grid pos
race_laps = st.sidebar.number_input("Race Laps (Scheduled)", value=50)

st.sidebar.markdown("---")
st.sidebar.markdown("**Simulation Conditions**")
altitude = st.sidebar.slider("Track Altitude (m)", 0, 2200, 10, help="Mexico City is high, Monaco is low.")
season_year = st.sidebar.number_input("Season Year", value=2025)

st.title("🏎️ Apex Strategy AI")
st.markdown("### Pre-Race Strategic Performance Predictor")

if st.button("RUN STRATEGY SIMULATION", type="primary", use_container_width=True):
    
    input_data = {
        'starting_position': [grid_position],
        'qualifying_pos': [qualifying_pos],
        'laps': [race_laps],
        'year': [season_year],
        'round': [1], # Generic round number
        'alt': [altitude],
        'constructor_name': [selected_team],
        'circuit_name': [selected_circuit]
    }
    
    df_input = pd.DataFrame(input_data)
    
    df_input = pd.get_dummies(df_input)
    
    df_input = df_input.reindex(columns=model_columns, fill_value=0)
    
    prediction = model.predict(df_input)[0]
    
    predicted_finish = grid_position - prediction
    
    predicted_finish = max(1.0, min(20.0, predicted_finish))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Starting Grid", value=f"P{grid_position}")
        
    with col2:
        delta_color = "normal"
        if prediction > 0.5: delta_color = "normal" # Gaining (positive delta in text, but model output logic varies)
        elif prediction < -0.5: delta_color = "inverse"
        
        st.metric(
            label="Predicted Finish", 
            value=f"P{predicted_finish:.1f}", 
            delta=f"{prediction:+.1f} Positions",
            delta_color="inverse" if prediction < 0 else "normal"
        )
        
    with col3:
        st.metric(label="Constructor Pace", value=selected_team)

    st.markdown("---")
    st.subheader("Strategy Visualization")
    
    chart_data = pd.DataFrame({
        'Position': [grid_position, predicted_finish],
        'Stage': ['Start', 'Finish']
    })
    
    st.bar_chart(chart_data.set_index('Stage'))
    
    if prediction > 1.0:
        st.success(f"**Strategy Insight:** The model predicts a **Recovery Drive**. The {selected_team} car has significant pace advantage over the cars immediately ahead.")
    elif prediction < -1.0:
        st.error(f"**Strategy Insight:** High Risk Detected. The model predicts a **Defensive Race**, likely due to qualifying higher than true race pace.")
    else:
        st.info(f"**Strategy Insight:** Stable Race Expected. Position retention is the primary goal.")

else:
    st.info("Adjust the race configuration in the sidebar and click 'Run Strategy Simulation' to generate insights.")

with st.expander("See Model Details"):
    st.write(f"Model Type: Random Forest Regressor")
    st.write(f"Input Features: {len(model_columns)} columns")