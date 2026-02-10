import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

## ---------------------------------------------------------
## 1. SETUP & CONFIGURATION
## ---------------------------------------------------------
st.set_page_config(
    page_title="Apex Strategy AI",
    page_icon="🏎️",
    layout="wide"
)

## Load Model
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('f1_position_model.pkl')
        model_columns = joblib.load('f1_model_columns.pkl')
        return model, model_columns
    except FileNotFoundError:
        return None, None

model, model_columns = load_artifacts()

if model is None:
    st.error("⚠️ Model files not found. Please run your notebook to generate .pkl files.")
    st.stop()

## ---------------------------------------------------------
## 2. HELPER FUNCTIONS
## ---------------------------------------------------------

def predict_position_change(driver_data):
    """
    Predicts the position change for a single row of driver data.
    """
    df_input = pd.DataFrame(driver_data)
    df_input = pd.get_dummies(df_input)
    df_input = df_input.reindex(columns=model_columns, fill_value=0)
    prediction = model.predict(df_input)[0]
    return prediction

def generate_virtual_grid(user_team, user_driver_pos):
    """
    Generates a full grid of 20 drivers.
    The user's driver is placed at 'user_driver_pos'.
    The rest are filled with 'AI Opponents' based on realistic 2024 team performance.
    """
    ## Realistic 2024 Grid Order (approximate strength)
    grid_template = [
        {"name": "Max Verstappen", "team": "Red Bull"},
        {"name": "Sergio Perez", "team": "Red Bull"},
        {"name": "Charles Leclerc", "team": "Ferrari"},
        {"name": "Carlos Sainz", "team": "Ferrari"},
        {"name": "Lando Norris", "team": "McLaren"},
        {"name": "Oscar Piastri", "team": "McLaren"},
        {"name": "Lewis Hamilton", "team": "Mercedes"},
        {"name": "George Russell", "team": "Mercedes"},
        {"name": "Fernando Alonso", "team": "Aston Martin"},
        {"name": "Lance Stroll", "team": "Aston Martin"},
        {"name": "Yuki Tsunoda", "team": "RB"},
        {"name": "Daniel Ricciardo", "team": "RB"},
        {"name": "Nico Hulkenberg", "team": "Haas F1 Team"},
        {"name": "Kevin Magnussen", "team": "Haas F1 Team"},
        {"name": "Alex Albon", "team": "Williams"},
        {"name": "Logan Sargeant", "team": "Williams"},
        {"name": "Esteban Ocon", "team": "Alpine"},
        {"name": "Pierre Gasly", "team": "Alpine"},
        {"name": "Valtteri Bottas", "team": "Kick Sauber"},
        {"name": "Zhou Guanyu", "team": "Kick Sauber"}
    ]
    
    ## 1. Create the Starting Grid List
    full_grid = []
    
    ## Place User's Driver
    user_driver = {"name": "USER (You)", "team": user_team, "start_pos": user_driver_pos}
    full_grid.append(user_driver)
    
    ## Fill the other 19 spots
    current_pos = 1
    for driver in grid_template:
        if len(full_grid) >= 20: break
        
        ## Skip the position taken by the user
        if current_pos == user_driver_pos:
            current_pos += 1
            
        ## Add opponent
        driver["start_pos"] = current_pos
        full_grid.append(driver)
        current_pos += 1
        
    ## Sort by Starting Position
    full_grid = sorted(full_grid, key=lambda x: x['start_pos'])
    
    return full_grid

## ---------------------------------------------------------
## 3. SIDEBAR: RACE SETTINGS
## ---------------------------------------------------------
st.sidebar.header("RACE CONFIGURATION")
st.sidebar.markdown("Configure the race conditions to simulate strategy.")

teams = ['Red Bull', 'Ferrari', 'Mercedes', 'McLaren', 'Aston Martin', 'Alpine', 'Williams', 'Haas F1 Team', 'Kick Sauber', 'RB']
circuits = ['Circuit de Monaco', 'Silverstone Circuit', 'Monza', 'Spa-Francorchamps', 'Marina Bay Street Circuit', 'Suzuka Circuit']

selected_team = st.sidebar.selectbox("Your Team", teams)
selected_circuit = st.sidebar.selectbox("Circuit", circuits)
grid_position = st.sidebar.slider("Starting Grid Position", 1, 20, 10)
race_laps = st.sidebar.number_input("Race Laps", value=50)

st.sidebar.markdown("---")
st.sidebar.markdown("**Track Conditions**")
altitude = st.sidebar.slider("Altitude (m)", 0, 2000, 10)
season_year = st.sidebar.selectbox("Season", [2024, 2025, 2026])

## ---------------------------------------------------------
## 4. MAIN DASHBOARD
## ---------------------------------------------------------

col_header_1, col_header_2 = st.columns([3, 1])
with col_header_1:
    st.title("🏎️ Apex Strategy AI")
    st.markdown(f"### Grand Prix Prediction: **{selected_circuit}**")
with col_header_2:
    ## VISUAL ELEMENT C: Track Map
    ## Logic: Checks if image exists, otherwise shows text
    image_path = f"tracks/{selected_circuit.lower().replace(' ', '_')}.png"
    if os.path.exists(image_path):
        st.image(image_path, caption="Track Layout")
    else:
        st.info("Track Map Loading...") 
        # (This is where your map would go if you download the images)

st.markdown("---")

if st.button("RUN STRATEGY SIMULATION", type="primary", use_container_width=True):
    
    ## 1. Generate the Virtual Grid (All 20 Drivers)
    virtual_grid = generate_virtual_grid(selected_team, grid_position)
    
    ## 2. Simulate Race for EVERY Driver
    simulation_results = []
    
    ## Create a progress bar for effect
    progress_bar = st.progress(0)
    
    for i, driver in enumerate(virtual_grid):
        ## Prepare Input for Model
        input_data = {
            'starting_position': [driver['start_pos']],
            'qualifying_pos': [driver['start_pos']], # Simplify for simulation
            'laps': [race_laps],
            'year': [season_year],
            'round': [1],
            'alt': [altitude],
            'constructor_name': [driver['team']],
            'circuit_name': [selected_circuit]
        }
        
        ## Predict
        pred_change = predict_position_change(input_data)
        
        ## Calculate Finish
        pred_finish = driver['start_pos'] - pred_change
        pred_finish = max(1.0, min(20.0, pred_finish)) # Clamp between 1 and 20
        
        simulation_results.append({
            "name": driver['name'],
            "team": driver['team'],
            "start": driver['start_pos'],
            "finish": pred_finish,
            "change": pred_change
        })
        
        ## Update progress
        progress_bar.progress((i + 1) / 20)
        
    progress_bar.empty() # Remove bar when done
    
    ## 3. Sort Results by Predicted Finish
    simulation_results = sorted(simulation_results, key=lambda x: x['finish'])
    
    ## Identify User's Result
    user_result = next(item for item in simulation_results if item["name"] == "USER (You)")
    
    ## ---------------------------------------------------------
    ## VISUAL ELEMENT A: MOVEMENT GAUGE
    ## ---------------------------------------------------------
    st.subheader("🏁 Race Outcome Prediction")
    
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        st.metric("Starting Position", f"P{user_result['start']}")
    
    with m2:
        ## Dynamic Color Logic
        delta_val = user_result['change']
        st.metric(
            "Predicted Finish", 
            f"P{int(round(user_result['finish']))}", 
            f"{delta_val:+.1f} Positions",
            delta_color="normal" # Green for positive (up), Red for negative (down)
        )
        
    with m3:
        st.metric("Overtake Probability", f"{abs(delta_val)*10:.1f}%", help="Estimated confidence")
        
    with m4:
        status = "Attack" if delta_val > 0.5 else "Defend" if delta_val < -0.5 else "Maintain"
        st.metric("Strategy Mode", status)

    ## ---------------------------------------------------------
    ## VISUAL ELEMENT B: VIRTUAL GRID VISUALIZER
    ## ---------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Live Leaderboard Simulation")
    
    ## Create two columns for the "Before vs After" look
    col_grid_start, col_arrow, col_grid_finish = st.columns([4, 1, 4])
    
    with col_grid_start:
        st.markdown("**STARTING GRID**")
        for driver in virtual_grid:
            ## Highlight the user
            bg_color = "#2e7bcf" if driver['name'] == "USER (You)" else "#262730"
            st.markdown(
                f"""<div style='background-color: {bg_color}; padding: 10px; border-radius: 5px; margin-bottom: 5px;'>
                    <b>P{driver['start_pos']}</b> - {driver['name']} <span style='color: #aaa; font-size: 0.8em;'>({driver['team']})</span>
                </div>""", 
                unsafe_allow_html=True
            )
            
    with col_arrow:
        st.markdown("<br>" * 5, unsafe_allow_html=True) # Spacer
        st.markdown("<h1 style='text-align: center; color: gray;'>➔</h1>", unsafe_allow_html=True)
        
    with col_grid_finish:
        st.markdown("**PREDICTED FINISH**")
        for i, driver in enumerate(simulation_results):
            final_pos = i + 1
            ## Calculate movement arrow
            pos_diff = driver['start'] - final_pos
            if pos_diff > 0: arrow = "🟢 ▲" # Gained places
            elif pos_diff < 0: arrow = "🔴 ▼" # Lost places
            else: arrow = "⚪ -"
            
            ## Highlight the user
            bg_color = "#2e7bcf" if driver['name'] == "USER (You)" else "#262730"
            
            st.markdown(
                f"""<div style='background-color: {bg_color}; padding: 10px; border-radius: 5px; margin-bottom: 5px;'>
                    <b>P{final_pos}</b> {arrow} {driver['name']}
                </div>""", 
                unsafe_allow_html=True
            )

else:
    st.info("👈 Configure your race strategy in the sidebar to begin.")