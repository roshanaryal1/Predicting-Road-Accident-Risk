# 🚗 Road Accident Risk Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://roadaccident-roshanar-aryal.streamlit.app)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An **AI-powered web application** that predicts road accident risk based on environmental and road conditions using machine learning. Built with Streamlit and Random Forest Regressor achieving **90.5% accuracy (R²)**.

---

## 🌐 Live Demo

**✨ Try it now:** [https://roadaccident-roshanar-aryal.streamlit.app](https://roadaccident-roshanar-aryal.streamlit.app)

![Status](https://img.shields.io/badge/Status-Live-success?style=for-the-badge)

---

## ✨ Features

### 🔮 **Risk Prediction**
- Real-time accident risk assessment
- Interactive input form with 12 key features
- Visual risk gauge with color-coded levels (LOW/MEDIUM/HIGH)
- AI-powered safety recommendations

### 📊 **Model Information**
- Detailed performance metrics
- Feature importance visualization
- Model statistics dashboard

### 🎨 **Modern UI/UX**
- Beautiful gradient design
- Responsive layout
- Smooth animations
- Mobile-friendly interface

---

## 🤖 Model Performance

| Metric | Score |
|--------|-------|
| **R² Score** | 90.5% |
| **Kaggle RMSE** | 0.05597 |
| **MAE** | 0.0398 |
| **Algorithm** | Random Forest Regressor |
| **Training Samples** | 517,754 |
| **Features** | 12 variables |

---

## 📋 Input Features

The model analyzes the following factors:

### 🛣️ Road Characteristics
- Road type (highway, urban, rural)
- Number of lanes (1-6)
- Road curvature (0-90°)
- Road signs presence

### 🌤️ Environmental Conditions
- Weather (clear, rain, fog, snow)
- Lighting conditions (daylight, dusk, darkness)
- Time of day (morning, afternoon, evening, night)

### 🚦 Traffic & Context
- Speed limit (20-130 km/h)
- Historical accident count
- Public road status
- Holiday status
- School season

---

## 🚀 Quick Start

### Option 1: Use the Live App (Recommended)
Simply visit: **[https://roadaccident-roshanar-aryal.streamlit.app](https://roadaccident-roshanar-aryal.streamlit.app)**

### Option 2: Run Locally

1. **Clone the repository**
```bash
git clone https://github.com/roshanaryal1/Predicting-Road-Accident-Risk.git
cd Predicting-Road-Accident-Risk
```

2. **Install dependencies**
```bash
pip install -r streamlit_requirements.txt
```

3. **Train the model** (first time only)
```bash
python train_and_save_model.py
```

4. **Run the app**
```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
Predicting-Road-Accident-Risk/
├── streamlit_app.py              # Main Streamlit application
├── train_and_save_model.py       # Model training script
├── streamlit_requirements.txt    # Python dependencies
├── data/
│   ├── train.csv                 # Training dataset
│   └── test.csv                  # Test dataset
├── model/                        # Trained model files (generated)
│   ├── accident_risk_model.pkl   # Random Forest model
│   └── label_encoders.pkl        # Categorical encoders
├── images/                       # Visualizations
│   ├── correlation_matrix.png
│   ├── feature_importance.png
│   └── model_comparison.png
└── README.md                     # This file
```

---

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **ML Library**: Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Deployment**: Streamlit Cloud
- **Version Control**: Git/GitHub

---

## 📊 Dataset

- **Source**: [Kaggle Playground Series S5E10](https://www.kaggle.com/competitions/playground-series-s5e10)
- **Training Samples**: 517,754
- **Features**: 12 variables
- **Target**: Accident risk probability (0-1)

---

## 🎯 Use Cases

- **Traffic Management**: Identify high-risk road conditions
- **Route Planning**: Choose safer travel routes
- **Urban Planning**: Improve road infrastructure
- **Insurance**: Risk assessment for policies
- **Education**: Road safety awareness
- **Research**: Traffic safety analysis

---

## 📈 Model Training

The Random Forest Regressor was trained using:

```python
# Key parameters
n_estimators = 100
max_depth = 20
random_state = 42

# Features used
- Road characteristics (4 features)
- Environmental conditions (4 features)
- Traffic & context factors (4 features)
```

**Training Process**:
1. Data preprocessing and cleaning
2. Label encoding for categorical variables
3. Train-test split (80-20)
4. Random Forest training
5. Model evaluation and validation
6. Model serialization

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 Future Enhancements

- [ ] Real-time weather API integration
- [ ] GPS location-based risk assessment
- [ ] Historical accident data visualization
- [ ] Multi-language support
- [ ] Mobile app version
- [ ] Advanced deep learning models
- [ ] Integration with navigation apps

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👨‍💻 Author

**Roshan Aryal**

- GitHub: [@roshanaryal1](https://github.com/roshanaryal1)
- LinkedIn: [Roshan Aryal](https://www.linkedin.com/in/roshanaryaal/)
- Email: roshanaryaal@gmail.com
- Portfolio: [roshanaryal.com](https://www.roshanaryal.com)
- Kaggle: [Competition Link](https://www.kaggle.com/competitions/playground-series-s5e10)

---

## 🙏 Acknowledgments

- Kaggle for providing the dataset
- Streamlit for the amazing framework
- Scikit-learn for ML tools
- The open-source community

---

## ⚠️ Disclaimer

This is a **predictive model for educational and research purposes**. While the model achieves high accuracy, it should not be the sole basis for critical safety decisions. Always:

- Follow local traffic laws and regulations
- Exercise caution while driving
- Consider multiple factors beyond model predictions
- Consult professional traffic safety experts for critical decisions

---

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/roshanaryal1/Predicting-Road-Accident-Risk/issues) page
2. Create a new issue with detailed description
3. Star ⭐ the repository if you find it helpful!

---

<div align="center">

**Made with ❤️ by Roshan Aryal**

[Live Demo](https://roadaccident-roshanar-aryal.streamlit.app) • [Report Bug](https://github.com/roshanaryal1/Predicting-Road-Accident-Risk/issues) • [Request Feature](https://github.com/roshanaryal1/Predicting-Road-Accident-Risk/issues)

</div>
