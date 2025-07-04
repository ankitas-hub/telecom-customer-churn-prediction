# Customer Churn Prediction & Analytics

Predicting customer churn is crucial for telecom companies to minimize revenue loss and improve customer satisfaction.  
This intermediate-level machine‑learning project walks through **exploratory data analysis (EDA)** and **supervised learning model development** using **NumPy, Pandas, Matplotlib, Seaborn, SciPy, and scikit‑learn**.

<div align="center">
  <img src="correlation_heatmap.png" alt="Correlation Heatmap" width="45%"/> &nbsp;
  <img src="churn_distribution.png" alt="Churn Distribution" width="45%"/>
</div>

## 📂 Repository Structure

```
customer-churn-prediction/
│
├── data/
│   └── telecom_customer_churn.csv         # Dataset
│
├── notebooks/
│   └── churn_analysis_and_prediction.ipynb # Full analysis & modelling notebook
│
├── churn_model.py                          # Script version of the notebook
├── requirements.txt                        # Python dependencies
├── README.md                               # You are here!
└── LICENSE
```

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/<your‑username>/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Create virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the notebook
jupyter notebook notebooks/churn_analysis_and_prediction.ipynb

# OR run the script directly
python churn_model.py
```

The script will generate:

* `correlation_heatmap.png`
* `churn_distribution.png`

and print detailed metrics for each model.

## 🧰 Features & Techniques

| Category | Details |
|----------|---------|
| **Data Analytics** | Descriptive stats, correlation heatmap, distribution plots |
| **Stat Tests** | Chi‑square test for categorical features vs. churn |
| **Models** | Logistic Regression, Random Forest, Support Vector Machine |
| **Evaluation** | Accuracy, Precision, Recall, F1‑score, Confusion Matrix |

## 📈 Results Snapshot

| Model | Accuracy (example) |
|-------|--------------------|
| Logistic Regression | ~0.82 |
| Random Forest | ~0.86 |
| SVM | ~0.79 |

*(Exact scores may vary due to the random synthetic dataset.)*

## 📝 Dataset

A **synthetic telecom customer churn dataset** (`telecom_customer_churn.csv`) with 1,000 samples and 21 features, generated to resemble common public churn datasets.

## 🏗️ Potential Improvements

* Hyper‑parameter tuning (GridSearchCV / RandomizedSearchCV)  
* Class imbalance handling (SMOTE / class weights)  
* Feature importance visualizations  
* Add Gradient Boosting or XGBoost models  
* Deploy as a REST API (FastAPI / Flask) or interactive dashboard (Streamlit)

## 🤝 Contributing

Feel free to submit issues or pull requests! For major changes, open a discussion first.

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.
