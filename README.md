# 🚗 Road Accident Risk Predictor

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33-ff69b4.svg)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/roshanaryal1/Predicting-Road-Accident-Risk?style=social)](https://github.com/roshanaryal1/Predicting-Road-Accident-Risk/stargazers)

> An AI-powered web application built with Streamlit that predicts road accident risk probability using environmental and road condition features.

---

## 🚀 Live Demo

The application is deployed and publicly accessible on Streamlit Community Cloud.

**[➡️ Try the Live Application Here](https://roadaccident-roshanar-aryal.streamlit.app/)**

*It's recommended to add a new screenshot of the final UI here.*

---

## ✨ Features

-   **🔮 Real-time Risk Prediction**: Instantly assess accident risk by inputting road, weather, and traffic conditions.
-   **📊 Interactive Gauge Chart**: Visualize the predicted risk probability on a color-coded gauge.
-   **💡 AI-Powered Recommendations**: Receive dynamic safety tips based on the provided conditions to mitigate risks.
-   **📈 Model Insights**: Explore the Random Forest model's performance metrics and the most influential features driving its predictions.
-   **🎨 Professional UI**: A modern, responsive, and visually appealing "glassmorphism" interface built for a great user experience.
-   **☁️ Auto-Training on Deploy**: The app automatically trains the ML model on the cloud server if it's not already present, solving deployment constraints for large model files.

---

## 🛠️ Tech Stack

-   **Web Framework**: [Streamlit](https://streamlit.io/)
-   **Machine Learning**: [Scikit-learn](https://scikit-learn.org/) (RandomForestRegressor)
-   **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
-   **Visualization**: [Plotly](https://plotly.com/)
-   **Deployment**: [Streamlit Community Cloud](https://streamlit.io/cloud)
-   **Version Control**: Git & GitHub

---

## 📁 Project Structure

```
road-accident-risk-predictor/
│
├── streamlit_app.py                # 🚀 Main Streamlit application
├── train_and_save_model.py         # 🤖 Script to train and save the ML model
│
├── model/                          # 💾 Saved model artifacts (generated on run)
│   ├── accident_risk_model.pkl
│   └── label_encoders.pkl
│
├── data/                           # 📁 Raw training and test data
│   ├── train.csv
│   └── test.csv
│
├── .python-version                 # 🐍 Specifies Python version for deployment
├── requirements.txt                # Python dependencies for local development
├── streamlit_requirements.txt      # Dependencies for Streamlit Cloud
├── packages.txt                    # System-level packages for Streamlit Cloud
├── README.md                       # This file
└── .gitignore                      # Specifies files for Git to ignore
```

---

## 🚀 Quick Start (Local Development)

Follow these steps to run the application on your local machine.

### 1. Prerequisites
-   Python 3.11+
-   `pip` package manager

### 2. Clone the Repository
```bash
git clone https://github.com/roshanaryal1/Predicting-Road-Accident-Risk.git
cd Predicting-Road-Accident-Risk
```

### 3. Set up a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.
```bash
# Create a virtual environment
python3 -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Model Training Script
The Streamlit app requires the model files to be present. Run the training script first to generate them.
```bash
python3 train_and_save_model.py
```
This will create the `model/` directory with `accident_risk_model.pkl` and `label_encoders.pkl`.

### 6. Launch the Streamlit App
```bash
streamlit run streamlit_app.py
```
✅ The application should now be running in your web browser!

---

## 👤 Author

**Roshan Aryal**

-   **GitHub**: [@roshanaryal1](https://github.com/roshanaryal1)
-   **LinkedIn**: [roshanaryaal](https://www.linkedin.com/in/roshanaryaal/)
-   **Portfolio**: [roshanaryal.com](https://www.roshanaryal.com)

---

## 📝 License

This project is licensed under the MIT License.
