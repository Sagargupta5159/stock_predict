# 📈 Stock Prediction App

An interactive web application for analyzing and forecasting stock prices using deep learning (LSTM). Built with **Streamlit**, **yfinance**, and **TensorFlow**.

![Stock Chart](https://img.shields.io/badge/Status-Active-brightgreen) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Streamlit](https://img.shields.io/badge/Framework-Streamlit-orange) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 🔍 Features

- 📊 Visualize historical stock data with interactive charts
- 📥 Fetch data from Yahoo Finance (`yfinance`)
- 🧠 Predict future stock prices using LSTM neural networks
- 💸 Simulate investment outcomes based on predicted trends
- 🎛️ Customize:
  - Stock symbol (e.g., AAPL, MSFT, TSLA)
  - Investment amount
  - Prediction period (days)

## 🚀 Live Demo

🔗 [Launch the App](https://stock-prediction-app-24-cs-ds-4b-14.streamlit.app/)*(Visit)*

## 🧰 Tech Stack

- **Frontend/UI**: Streamlit
- **Backend**: Python
- **Machine Learning**: TensorFlow (LSTM)
- **Data Source**: Yahoo Finance via `yfinance`
- **Visualization**: Plotly

## 📸 Screenshots

> *(<img width="1919" height="854" alt="Screenshot 2025-07-28 010234" src="https://github.com/user-attachments/assets/2b1cc41b-902e-411f-8af2-8bd4d2c54787" />
)*
> <img width="1919" height="889" alt="Screenshot 2025-07-28 010249" src="https://github.com/user-attachments/assets/aadada1b-7b96-4774-910d-e6f39009f9d4" />
<img width="1919" height="904" alt="Screenshot 2025-07-28 010310" src="https://github.com/user-attachments/assets/4e0e112f-5724-4b7c-ba91-03a27de3aea6" />



## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/stock-prediction-app.git
cd stock-prediction-app


### 2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt

3. Run the app locally
streamlit run app.py


📦 Requirements
See requirements.txt
streamlit
yfinance
pandas
plotly
scikit-learn
tensorflow
numpy


📁 Project Structure
📦 stock-prediction-app
├── app.py                  # Main Streamlit app
├── model.py                # LSTM model definition (if separated)
├── utils.py                # Utility functions (e.g., preprocessing)
├── requirements.txt
└── README.md

🤝 Contributing
Contributions are welcome!
Fork this repo
Create a feature branch
Submit a pull request

📜 License
This project is licensed under the MIT License.
See LICENSE for details.

⭐ Show your support
If you find this project helpful, please ⭐ star the repo and share it with others!
markdown
Copy
Edit
