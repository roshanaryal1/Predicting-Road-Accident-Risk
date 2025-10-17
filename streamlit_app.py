"""
Streamlit Web App for Road Accident Risk Predictor
Simple, fast deployment - no backend setup needed!
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Road Accident Risk Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional, visually appealing UI
st.markdown("""
    <style>
    /* --- General --- */
    body {
        background-color: #0e1117;
    }
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #fafafa;
    }
    .stApp > header {
        background-color: transparent;
    }
    
    /* --- Sidebar --- */
    [data-testid="stSidebar"] {
        background: rgba(38, 43, 56, 0.6);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    [data-testid="stSidebar"] .css-1d391kg {
        color: #fafafa;
    }
    [data-testid="stSidebar"] .stRadio > label {
        font-size: 1.1rem;
        font-weight: 600;
    }

    /* --- Main Content --- */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* --- Glassmorphism Card Effect --- */
    .glass-card {
        background: rgba(38, 43, 56, 0.6);
        backdrop-filter: blur(10px);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }

    /* --- Input Widgets --- */
    .stSelectbox, .stSlider, .stNumberInput, .stCheckbox {
        margin-bottom: 10px;
    }

    /* --- Predict Button --- */
    .stButton>button {
        background: linear-gradient(90deg, #6A82FB, #FC5C7D);
        color: white;
        font-weight: bold;
        border-radius: 12px;
        padding: 12px 36px;
        border: none;
        width: 100%;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
        font-size: 18px;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        background: linear-gradient(90deg, #FC5C7D, #6A82FB);
    }
    
    /* --- Result Display --- */
    .result-card {
        background: rgba(38, 43, 56, 0.8);
        padding: 30px; 
        border-radius: 15px; 
        border-left: 5px solid;
        text-align: center;
    }
    
    /* --- Hide Streamlit Branding --- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    """Load trained model, encoders, and feature order"""
    try:
        model = joblib.load('model/accident_risk_model.pkl')
        encoders = joblib.load('model/label_encoders.pkl')
        feature_order = joblib.load('model/feature_order.pkl') # Load the feature order
        return model, encoders, feature_order
    except FileNotFoundError:
        # Auto-train model silently (for Streamlit Cloud deployment)
        try:
            import subprocess
            import sys
            with st.spinner("🤖 Initializing AI model... This may take a moment..."):
                result = subprocess.run([sys.executable, 'train_and_save_model.py'], 
                                      capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                # Try loading again after training
                model = joblib.load('model/accident_risk_model.pkl')
                encoders = joblib.load('model/label_encoders.pkl')
                feature_order = joblib.load('model/feature_order.pkl')
                st.rerun()  # Silently reload the page
                return model, encoders, feature_order
            else:
                st.error(f"❌ Unable to initialize model. Details: {result.stderr}")
                return None, None, None
        except Exception as e:
            st.error(f"❌ A system error occurred during model initialization: {e}")
            return None, None, None

def get_risk_level(risk_score):
    """Categorize risk score"""
    if risk_score < 0.33:
        return "LOW", "🟢", "#4CAF50"
    elif risk_score < 0.67:
        return "MEDIUM", "🟡", "#FF9800"
    else:
        return "HIGH", "🔴", "#F44336"

def create_gauge_chart(risk_score):
    """Create gauge chart for risk visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Score (%)", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 33], 'color': '#90EE90'},
                {'range': [33, 67], 'color': '#FFD700'},
                {'range': [67, 100], 'color': '#FF6B6B'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def generate_recommendations(data, risk_score):
    """Generate safety recommendations"""
    recommendations = []
    
    if risk_score > 0.67:
        recommendations.append('🚨 HIGH RISK: Extreme caution required!')
    
    if data.get('weather') in ['rain', 'fog', 'snow']:
        recommendations.append('🌧️ Poor weather - reduce speed by 20-30%')
    
    if 'darkness' in data.get('lighting', '').lower():
        recommendations.append('🔦 Low visibility - use headlights')
    
    if data.get('speed_limit', 0) > 80:
        recommendations.append('⚡ High speed zone - maintain safe distance')
    
    if data.get('num_accidents', 0) > 10:
        recommendations.append('⚠️ High accident history area')
    
    if data.get('curvature', 0) > 45:
        recommendations.append('🔄 Sharp curves - reduce speed')
    
    if not recommendations or risk_score < 0.33:
        recommendations.append('✅ Conditions are relatively safe')
    
    return recommendations

def main():
    # Header
    st.markdown("<h1 style='text-align: center;'>🚗 Road Accident Risk Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.1rem; color: #a0a0a0;'>An AI-Powered Assessment System for Proactive Road Safety</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model
    model, encoders, feature_order = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.image("https://i.imgur.com/M2G5pB4.png", width=100)
        st.title("Navigation")
        page = st.radio("Select Page", ["🔮 Predict Risk", "📈 Model Info", "ℹ️ About"], label_visibility="collapsed")
        
        st.markdown("---")
        st.markdown("### 🎯 Quick Stats")
        st.markdown("""
            <div class="glass-card" style="padding: 15px;">
            <p style="margin: 0; font-weight: bold;">Model: <span style="color: #6A82FB;">Random Forest</span></p>
            <p style="margin: 0; font-weight: bold;">Accuracy (R²): <span style="color: #6A82FB;">90.5%</span></p>
            <p style="margin: 0; font-weight: bold;">Features: <span style="color: #6A82FB;">12</span></p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.info("This app predicts accident risk probability. Always drive safely.")

    # Main content
    if page == "🔮 Predict Risk":
        show_prediction_page(model, encoders, feature_order)
    elif page == "📈 Model Info":
        show_model_info(model, feature_order)
    else:
        show_about()

def show_prediction_page(model, encoders, feature_order):
    """Prediction interface"""
    st.header("🔮 Predict Accident Risk")
    st.markdown("<p style='color: #a0a0a0;'>Enter road and environmental conditions to assess accident risk probability.</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("<h5>🛣️ Road Characteristics</h5>", unsafe_allow_html=True)
        road_type = st.selectbox("Road Type", ['highway', 'urban', 'rural'], help="Select the type of road (e.g., highway, urban street).")
        num_lanes = st.slider("Number of Lanes", 1, 6, 2, help="Total number of lanes on the road.")
        curvature = st.slider("Road Curvature (°)", 0, 90, 10, help="Estimated degree of road curvature. Higher values mean sharper curves.")
        road_signs = st.checkbox("Road Signs Present", value=True, help="Check if warning or informational signs are present.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("<h5>🌤️ Environmental</h5>", unsafe_allow_html=True)
        weather = st.selectbox("Weather", ['clear', 'rain', 'fog', 'snow'], help="Current weather conditions.")
        lighting = st.selectbox("Lighting", ['daylight', 'dawn/dusk', 'darkness', 'darkness_with_lights'], help="Current lighting conditions.")
        time_of_day = st.selectbox("Time of Day", ['morning', 'afternoon', 'evening', 'night'], help="Select the current time period.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("<h5>🚦 Traffic & Context</h5>", unsafe_allow_html=True)
        speed_limit = st.slider("Speed Limit (km/h)", 20, 130, 60, step=10, help="Posted speed limit for the area.")
        num_accidents = st.number_input("Historical Accidents", 0, 100, 5, help="Number of previously reported accidents at this location.")
        public_road = st.checkbox("Public Road", value=True, help="Check if this is a publicly maintained road.")
        holiday = st.checkbox("Holiday", value=False, help="Check if today is a public holiday.")
        school_season = st.checkbox("School Season", value=True, help="Check if it's currently a school season.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🔍 Predict Accident Risk"):
        # Prepare data in the correct order
        input_data = pd.DataFrame({
            'road_type': [road_type],
            'num_lanes': [num_lanes],
            'lighting': [lighting],
            'weather': [weather],
            'curvature': [curvature],
            'speed_limit': [speed_limit],
            'road_signs_present': [road_signs],
            'num_reported_accidents': [num_accidents],
            'time_of_day': [time_of_day],
            'public_road': [public_road],
            'holiday': [holiday],
            'school_season': [school_season]
        })
        
        # Ensure the DataFrame has the correct column order using the loaded list
        input_data = input_data[feature_order]
        
        # Encode categorical variables
        categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
        for col in categorical_cols:
            if col in encoders:
                input_data[col] = encoders[col].transform(input_data[col].astype(str))
        
        # Predict
        with st.spinner("🤖 Analyzing road conditions..."):
            prediction = model.predict(input_data)[0]
            risk_level, emoji, color = get_risk_level(prediction)
        
        # Display results
        st.markdown("---")
        st.header("📊 Prediction Result")
        
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.markdown(f"""
                <div class="result-card" style='border-left-color: {color};'>
                    <h3 style='color: {color}; margin-bottom: 10px;'>{risk_level} RISK</h3>
                    <h1 style='font-size: 4rem; margin: 0;'>{prediction*100:.1f}%</h1>
                    <p style='color: #a0a0a0; margin-top: 5px;'>Risk Probability</p>
                    <p style='color: #a0a0a0; font-size: 0.9rem;'>(Score: {prediction:.4f})</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            fig = create_gauge_chart(prediction)
            st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("---")
        st.markdown("### 💡 Safety Recommendations")
        data_dict = {
            'weather': weather,
            'lighting': lighting,
            'speed_limit': speed_limit,
            'num_accidents': num_accidents,
            'curvature': curvature
        }
        recommendations = generate_recommendations(data_dict, prediction)
        
        for rec in recommendations:
            st.info(rec)

def show_model_info(model, feature_order):
    """Model information page"""
    st.header("📈 Model Information")
    st.markdown("<p style='color: #a0a0a0;'>Explore the AI model's performance metrics and feature analysis.</p>", unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🤖 Model Details")
        st.markdown("""
        **Algorithm**: Random Forest Regressor
        
        **Training Data**: 517,754 samples
        
        **Performance**:
        - R² Score: 90.5%
        - Kaggle RMSE: 0.05597
        - MAE: 0.0398
        
        **Features**: 12 key variables used for prediction.
        """)
    
    with col2:
        st.markdown("<h5>🎯 Feature Importance</h5>", unsafe_allow_html=True)
        
        if hasattr(model, 'feature_importances_'):
            # Use the loaded feature_order for consistency
            features = feature_order
            importances = model.feature_importances_
            
            df = pd.DataFrame({
                'Feature': features,
                'Importance': importances * 100
            }).sort_values('Importance', ascending=False).head(10)
            
            st.bar_chart(df.set_index('Feature'))
    st.markdown("</div>", unsafe_allow_html=True)

def show_about():
    """About page"""
    st.header("ℹ️ About This Project")
    st.markdown("<p style='color: #a0a0a0;'>Learn more about the technology and purpose behind this application.</p>", unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("""
    ### 🚗 Road Accident Risk Predictor
    
    This application uses **machine learning** to predict road accident risk 
    based on environmental and road condition factors.
    
    #### 🎯 Purpose
    - Identify high-risk road scenarios
    - Improve road safety awareness
    - Assist in traffic management
    - Provide data-driven safety recommendations
    
    #### 🛠️ Technology
    - **Framework**: Streamlit
    - **ML Model**: Random Forest Regressor
    - **Libraries**: Scikit-learn, Pandas, NumPy, Plotly
    
    #### 📊 Dataset
    - **Source**: Kaggle Playground Series S5E10
    - **Training Data**: 517,754 samples
    - **Features**: 12 variables
    
    #### 🔗 Important Links
    - **GitHub Repository**: [Click here](https://github.com/roshanaryal1/Predicting-Road-Accident-Risk)
    - **Kaggle Competition**: [Click here](https://www.kaggle.com/competitions/playground-series-s5e10)
    
    ---
    
    **⚠️ Disclaimer**: This is a predictive model developed for educational and demonstrative purposes. Always adhere to local traffic laws and exercise caution while driving.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
