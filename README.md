📊 Customer Churn Prediction Web App

This project is an end-to-end Customer Churn Prediction System built using Machine Learning (ANN) and deployed as an interactive Streamlit web application. The model predicts whether a banking customer is likely to stay or exit, helping businesses take proactive retention actions.

🚀 Features

🔹 Artificial Neural Network (ANN)–based prediction model

🔹 Real-time churn prediction through Streamlit

🔹 Clean and responsive user interface

🔹 StandardScaler used for feature preprocessing

🔹 Accepts customer inputs like age, credit score, balance, geography, etc.

🔹 Displays churn probability and prediction outcome

🧠 Model Details

Architecture: ANN with Dense layers

Framework: TensorFlow / Keras

Dataset Size: 15,000+ customers

Performance: ~79% accuracy on test data

Preprocessing: Balanced dataset + StandardScaler

📂 Project Structure
├── app.py                  # Streamlit web app
├── customer_churn_model.h5 # Trained ANN model
├── scaler.pkl              # StandardScaler object
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation

💡 How to Run the App Locally
pip install -r requirements.txt
streamlit run app.py

🌐 Live App (Streamlit Cloud)

👉 Link will appear here once deployed

📌 Use Cases

Banking customer retention

Telecom churn prediction

Subscription service churn analysis

Marketing decision support

Customer segmentation and risk profiling

🤝 Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

📬 Contact

For queries or collaboration, feel free to reach out.
