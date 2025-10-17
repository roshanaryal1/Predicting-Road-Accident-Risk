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

# Custom CSS for beautiful UI
st.markdown("""
    <style>
    /* Main background gradient */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Enhanced button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 12px;
        padding: 12px 36px;
        border: none;
        width: 100%;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        font-size: 16px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Input field styling */
    .stSelectbox, .stSlider, .stNumberInput {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 5px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] .sidebar-content {
        background: rgba(255, 255, 255, 0.1);
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Header styling */
    h1, h2, h3 {
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Info box styling */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid #667eea;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Smooth animations */
    * {
        transition: all 0.3s ease;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    """Load trained model and encoders"""
    try:
        model = joblib.load('model/accident_risk_model.pkl')
        encoders = joblib.load('model/label_encoders.pkl')
        return model, encoders
    except FileNotFoundError:
        # Auto-train model silently (for Streamlit Cloud deployment)
        try:
            import subprocess
            import sys
            with st.spinner("🤖 Initializing AI model... Please wait..."):
                result = subprocess.run([sys.executable, 'train_and_save_model.py'], 
                                      capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                model = joblib.load('model/accident_risk_model.pkl')
                encoders = joblib.load('model/label_encoders.pkl')
                st.rerun()  # Silently reload the page
                return model, encoders
            else:
                st.error(f"❌ Unable to initialize model. Please try again later.")
                return None, None
        except Exception as e:
            st.error(f"❌ System error. Please refresh the page.")
            return None, None

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
    st.markdown("<h1 style='text-align: center; color: white;'>🚗 Road Accident Risk Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px; color: white;'>AI-Powered Risk Assessment System</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model
    model, encoders = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/road.png", width=100)
        st.title("📊 Navigation")
        page = st.radio("Select Page", ["🔮 Predict Risk", "📈 Model Info", "ℹ️ About"])
        
        st.markdown("---")
        st.markdown("### 🎯 Quick Stats")
        st.info("**Model**: Random Forest")
        st.success("**Accuracy**: 90.5% (R²)")
        st.warning("**Features**: 12 variables")
    
    # Main content
    if page == "🔮 Predict Risk":
        show_prediction_page(model, encoders)
    elif page == "📈 Model Info":
        show_model_info(model)
    else:
        show_about()

def show_prediction_page(model, encoders):
    """Prediction interface"""
    st.markdown("<h1 style='text-align: center; font-size: 2.5rem; margin-bottom: 30px;'>🔮 Predict Accident Risk</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.8); margin-bottom: 40px;'>Enter road and environmental conditions to assess accident risk probability</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🛣️ Road Characteristics")
        st.markdown("<div style='background: rgba(255,255,255,0.1); padding: 20px; border-radius: 12px; backdrop-filter: blur(10px);'>", unsafe_allow_html=True)
        road_type = st.selectbox("Road Type", ['highway', 'urban', 'rural'], help="Select the type of road")
        num_lanes = st.slider("Number of Lanes", 1, 6, 2, help="Total lanes on the road")
        curvature = st.slider("Road Curvature (°)", 0, 90, 10, help="Degree of road curvature")
        road_signs = st.checkbox("Road Signs Present", value=True, help="Are warning signs present?")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🌤️ Environmental")
        st.markdown("<div style='background: rgba(255,255,255,0.1); padding: 20px; border-radius: 12px; backdrop-filter: blur(10px);'>", unsafe_allow_html=True)
        weather = st.selectbox("Weather", ['clear', 'rain', 'fog', 'snow'], help="Current weather conditions")
        lighting = st.selectbox("Lighting", ['daylight', 'dawn/dusk', 'darkness', 'darkness_with_lights'], help="Light conditions")
        time_of_day = st.selectbox("Time of Day", ['morning', 'afternoon', 'evening', 'night'], help="Current time period")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("### 🚦 Traffic & Context")
        st.markdown("<div style='background: rgba(255,255,255,0.1); padding: 20px; border-radius: 12px; backdrop-filter: blur(10px);'>", unsafe_allow_html=True)
        speed_limit = st.slider("Speed Limit (km/h)", 20, 130, 60, step=10, help="Posted speed limit")
        num_accidents = st.number_input("Historical Accidents", 0, 100, 5, help="Past accidents at location")
        public_road = st.checkbox("Public Road", value=True, help="Is this a public road?")
        holiday = st.checkbox("Holiday", value=False, help="Is today a holiday?")
        school_season = st.checkbox("School Season", value=True, help="Is school in session?")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button("🔍 Predict Accident Risk", use_container_width=True):
        # Prepare data
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
        st.success("✅ Prediction Complete!")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                            padding: 30px; border-radius: 15px; border: 4px solid {color};'>
                    <h1 style='text-align: center; font-size: 4rem;'>{emoji}</h1>
                    <h2 style='text-align: center; color: {color};'>{risk_level} RISK</h2>
                    <h1 style='text-align: center; font-size: 3rem;'>{prediction*100:.2f}%</h1>
                    <p style='text-align: center; color: #666;'>Risk Probability</p>
                    <p style='text-align: center; color: #666;'>Score: {prediction:.4f}</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            fig = create_gauge_chart(prediction)
            st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
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

def show_model_info(model):
    """Model information page"""
    st.markdown("<h1 style='text-align: center; font-size: 2.5rem; margin-bottom: 30px;'>📈 Model Information</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.8); margin-bottom: 40px;'>Explore the AI model's performance metrics and feature analysis</p>", unsafe_allow_html=True)
    
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
        
        **Features**: 12 key variables
        """)
    
    with col2:
        st.markdown("### 🎯 Feature Importance")
        
        if hasattr(model, 'feature_importances_'):
            features = ['curvature', 'lighting', 'speed_limit', 'weather', 
                       'num_reported_accidents', 'num_lanes', 'time_of_day', 
                       'road_type', 'public_road', 'holiday', 'road_signs_present', 
                       'school_season']
            importances = model.feature_importances_
            
            df = pd.DataFrame({
                'Feature': features,
                'Importance': importances * 100
            }).sort_values('Importance', ascending=False).head(10)
            
            st.bar_chart(df.set_index('Feature'))

def show_about():
    """About page"""
    st.markdown("<h1 style='text-align: center; font-size: 2.5rem; margin-bottom: 30px;'>ℹ️ About This Project</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.8); margin-bottom: 40px;'>Learn more about the technology and purpose behind this application</p>", unsafe_allow_html=True)
    
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
    - **Training**: 517,754 samples
    - **Features**: 12 variables
    
    #### 🔗 Links
    - [GitHub Repository](https://github.com/roshanaryal1/Predicting-Road-Accident-Risk)
    - [Kaggle Competition](https://www.kaggle.com/competitions/playground-series-s5e10)
    
    ---
    
    **⚠️ Disclaimer**: This is a predictive model for educational purposes. 
    Always follow traffic laws and exercise caution while driving.
    """)

if __name__ == "__main__":
    main()
