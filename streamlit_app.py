import streamlit as st
import pandas as pd
import joblib
import numpy as np

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
    st.error("⚠️ Model files not found. Please run your notebook first to generate .pkl files.")
    st.stop()

## ---------------------------------------------------------
## 2. HELPER DATA (To make the app look real)
## ---------------------------------------------------------

## Real Drivers for 2025 (Approximation for the demo)
TEAM_DRIVERS = {
    'Red Bull': ['Max Verstappen', 'Sergio Perez'],
    'Ferrari': ['Charles Leclerc', 'Lewis Hamilton'],
    'Mercedes': ['George Russell', 'Andrea Kimi Antonelli'],
    'McLaren': ['Lando Norris', 'Oscar Piastri'],
    'Aston Martin': ['Fernando Alonso', 'Lance Stroll'],
    'Alpine': ['Pierre Gasly', 'Jack Doohan'],
    'Williams': ['Alex Albon', 'Carlos Sainz'],
    'RB': ['Yuki Tsunoda', 'Liam Lawson'],
    'Kick Sauber': ['Nico Hulkenberg', 'Gabriel Bortoleto'],
    'Haas F1 Team': ['Esteban Ocon', 'Oliver Bearman']
}

## Map URLs for Visual Element C
TRACK_MAPS = {
    'Circuit de Monaco': 'https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Circuit_de_Monaco_2003-2014.png/800px-Circuit_de_Monaco_2003-2014.png',
    'Silverstone Circuit': 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Silverstone_Circuit_2020.png/800px-Silverstone_Circuit_2020.png',
    'Monza': 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/b8/Monza_track_map.svg/800px-Monza_track_map.svg.png',
    'Spa-Francorchamps': 'https://upload.wikimedia.org/wikipedia/commons/thumb/5/54/Spa-Francorchamps_of_Belgium.svg/800px-Spa-Francorchamps_of_Belgium.svg.png',
    'Marina Bay Street Circuit': 'https://upload.wikimedia.org/wikipedia/commons/thumb/5/51/Marina_Bay_Street_Circuit_2023.svg/800px-Marina_Bay_Street_Circuit_2023.svg.png',
    'Suzuka Circuit': 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/Suzuka_circuit_map_2005.svg/800px-Suzuka_circuit_map_2005.svg.png'
}

## ---------------------------------------------------------
## 3. SIDEBAR INPUTS
## ---------------------------------------------------------
st.sidebar.header("STRATEGY CONFIGURATION")
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg", width=100)

selected_team = st.sidebar.selectbox("Your Team", list(TEAM_DRIVERS.keys()))
selected_driver = st.sidebar.selectbox("Your Driver", TEAM_DRIVERS[selected_team])
selected_circuit = st.sidebar.selectbox("Select Circuit", list(TRACK_MAPS.keys()))

st.sidebar.subheader("Race Conditions")
grid_position = st.sidebar.slider("Starting Grid Position", 1, 20, 10)
qualifying_pos = st.sidebar.slider("Qualifying Pace (P1-P20)", 1, 20, grid_position)
race_laps = st.sidebar.number_input("Race Laps", value=50)
altitude = st.sidebar.slider("Altitude (m)", 0, 2200, 10)
season_year = st.sidebar.number_input("Season", value=2025)

## ---------------------------------------------------------
## 4. MAIN DASHBOARD
## ---------------------------------------------------------

st.title("🏎️ Apex Strategy AI")
st.caption("Real-time Strategic Performance Prediction Engine")

if st.button("RUN RACE SIMULATION", type="primary", use_container_width=True):
    
    ## -----------------------------------------------------
    ## STEP A: PREDICT USER RESULT
    ## -----------------------------------------------------
    
    ## Prepare Input
    input_df = pd.DataFrame({
        'starting_position': [grid_position],
        'qualifying_pos': [qualifying_pos],
        'laps': [race_laps],
        'year': [season_year],
        'round': [1], 'alt': [altitude],
        'constructor_name': [selected_team],
        'circuit_name': [selected_circuit]
    })
    
    ## Encode
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    
    ## Predict
    user_pred_change = model.predict(input_df)[0]
    user_finish = grid_position - user_pred_change
    user_finish = max(1, min(20, user_finish)) ## Clamp between 1 and 20
    
    ## -----------------------------------------------------
    ## STEP B: VISUAL ELEMENT A (MOVEMENT GAUGE)
    ## -----------------------------------------------------
    
    st.markdown("### 📊 Strategy Forecast")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Grid Start", f"P{grid_position}")
    
    with col2:
        ## Color Logic for Element A
        st.metric(
            "Predicted Finish", 
            f"P{user_finish:.0f}", 
            f"{user_pred_change:+.1f} Places",
            delta_color="normal" ## Green = Good (Positive number means gained places)
        )
        
    with col3:
        st.metric("Track Temp", "28°C", "Dry") ## Static context
        
    with col4:
        risk_level = "HIGH" if user_pred_change < -2 else "LOW"
        st.metric("Risk Level", risk_level, "AI Assessment", delta_color="off")

    st.markdown("---")

    ## -----------------------------------------------------
    ## STEP C: VISUAL ELEMENT B (VIRTUAL GRID VISUALIZER)
    ## -----------------------------------------------------
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader(f"📍 Track Context: {selected_circuit}")
        ## Visual Element C: Map
        if selected_circuit in TRACK_MAPS:
            st.image(TRACK_MAPS[selected_circuit], caption="Circuit Layout", use_container_width=True)
        else:
            st.info("Map unavailable for this track.")
            
    with col_right:
        st.subheader("🏁 Predicted Leaderboard")
        
        ## 1. GENERATE A FULL GRID (SIMULATION)
        ## We create a fake grid of 20 drivers to make the app look "Live"
        grid_data = []
        
        ## We just assign random grid slots to other drivers for the demo
        ## In a real app, you'd let the user configure the whole grid
        current_grid = 1
        
        for team, drivers in TEAM_DRIVERS.items():
            for driver in drivers:
                ## If this is the USER's driver, use their inputs
                if driver == selected_driver:
                    row = {
                        'Driver': driver, 'Team': team, 'Start': grid_position,
                        'Pred_Change': user_pred_change, 'Finish': user_finish
                    }
                else:
                    ## For AI drivers, we assign a grid slot and run a quick prediction
                    ## Skip the user's grid slot
                    if current_grid == grid_position: current_grid += 1
                    
                    ai_start = current_grid
                    ## Mock AI Prediction (We use the model!)
                    ai_input = input_df.copy()
                    ai_input['starting_position'] = ai_start
                    ## We assume opponents qualify where they start
                    ai_input['qualifying_pos'] = ai_start 
                    
                    ## Handle Team Name OHE for AI
                    ai_input = ai_input.reindex(columns=model_columns, fill_value=0)
                    if f'constructor_name_{team}' in model_columns:
                        ai_input[f'constructor_name_{team}'] = 1
                    
                    ai_pred_change = model.predict(ai_input)[0]
                    ai_finish = max(1, min(20, ai_start - ai_pred_change))
                    
                    row = {
                        'Driver': driver, 'Team': team, 'Start': ai_start,
                        'Pred_Change': ai_pred_change, 'Finish': ai_finish
                    }
                    current_grid += 1
                
                grid_data.append(row)
        
        ## 2. SORT BY PREDICTED FINISH
        results_df = pd.DataFrame(grid_data)
        results_df = results_df.sort_values(by='Finish', ascending=True)
        results_df = results_df.reset_index(drop=True)
        
        ## 3. DISPLAY AS A TABLE WITH ARROWS
        ## We iterate through the top 10 to display nicely
        st.write("Top 10 Predicted Finishers")
        
        for i, row in results_df.head(10).iterrows():
            pos = i + 1
            driver = row['Driver']
            team = row['Team']
            change = row['Pred_Change']
            
            ## Arrow Logic
            if change > 0.5: icon = "🟢 ▲"  # Gained
            elif change < -0.5: icon = "🔴 ▼" # Lost
            else: icon = "⚪ -" # Stable
            
            ## Highlight the User
            if driver == selected_driver:
                st.markdown(f"**{pos}. {driver} ({team})** {icon} {abs(change):.1f}")
            else:
                st.markdown(f"{pos}. {driver} ({team}) {icon} {abs(change):.1f}")

else:
    st.info("👈 Configure race parameters in the sidebar and click RUN to initialize strategy engine.")