
## Usage Instructions

### Requirements
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Required Python packages:
	- pandas
	- numpy
	- matplotlib
	- seaborn
	- scikit-learn
	- statsmodels

Install dependencies with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```

### Running the Notebooks
1. Open any of the model notebooks (`Linear_Regression_Model.ipynb`, `SARIMA_Model.ipynb`, `Random_Forest_Model.ipynb`) in Jupyter.
2. Set `DATASET_INDEX` at the top of the notebook to select which dataset to use (1–5).
3. Run all cells to train the model and generate forecasts.
4. Outputs and metrics are saved in the corresponding output folders (`lr_outputs/`, `sarima_outputs/`, `rf_outputs/`).
5. After running all three model notebooks, open `Model_Comparison.ipynb` to view the comparative analysis.

### Notes
- All notebooks are self-contained and can be run independently.
- Datasets are located in `final_msme_datasets/`.
- Forecast horizon and other parameters can be adjusted at the top of each notebook.
Applies a Random Forest regressor to predict income and expenses based on operational and economic features. The notebook includes feature importance analysis and provides actionable recommendations. Model performance is measured with MAE, RMSE, and R², and results are saved for comparison.

### Model Comparison
Aggregates the saved metrics from all three models (Linear Regression, SARIMA, Random Forest) and presents a side-by-side comparison. The notebook displays a summary table of MAE, RMSE, and R² for each model and target (income, expenses), and visualizes the results with bar plots for easy interpretation.
# MSME Financial Forecasting Project

This project provides a comprehensive forecasting solution for Micro, Small, and Medium Enterprises (MSMEs) using real-world financial datasets. The goal is to predict future income, expenses, and cash flow using multiple machine learning and statistical models, enabling better financial planning and risk management for MSMEs.

## Project Structure

- **Linear_Regression_Model.ipynb**: Linear regression forecasting notebook
- **SARIMA_Model.ipynb**: SARIMA (Seasonal ARIMA) time series forecasting notebook
- **Random_Forest_Model.ipynb**: Random Forest regression forecasting notebook
- **Model_Comparison.ipynb**: Compares the performance of all models
- **final_msme_datasets/**: Contains five CSV files with monthly MSME financial data

## Dataset Description

Each CSV in `final_msme_datasets/` contains monthly records for an MSME, with the following columns:

- `Month-Year`: Month and year (YYYY-MM)
- `Sales Revenue (₹)`: Revenue from sales
- `Service Fees (₹)`: Revenue from services
- `Rent (₹)`, `Utilities (₹)`, `Salaries & Wages (₹)`, `Raw Materials / Inventory (₹)`, `Transportation / Logistics (₹)`, `Loan Repayments & Interest (₹)`: Major expense categories
- `Number of Orders`, `Customers`, `Average Order Value`: Operational metrics
- `Seasonality_Flag`: 1 if month is seasonal, 0 otherwise
- `Fuel Price Index`: External economic factor
- `Total Income (₹)`: Sum of sales and service fees
- `Total Expenses (₹)`: Sum of all expense categories
- `Net Cash Flow (₹)`: Total Income minus Total Expenses

Each dataset covers multiple years, providing a rich time series for model training and evaluation.

## Model Notebooks

### Linear Regression Model
Implements a multivariate linear regression to forecast future income, expenses, and cash flow. The notebook preprocesses the data, splits into train/test sets, fits linear models, and evaluates performance using MAE, RMSE, and R². Outputs include predicted values and alert flags for negative cash flow.

### SARIMA Model
Uses the SARIMA (Seasonal ARIMA) time series model to capture seasonality and trends in the financial data. The notebook fits SARIMAX models for income and expenses, forecasts future values, and evaluates with MAE, RMSE, and R². Outputs are saved for comparison.

### Random Forest Model
Applies a Random Forest regressor to predict income and expenses based on operational and economic features. The notebook includes feature importance analysis and provides actionable recommendations. Model performance is measured with MAE, RMSE, and R², and results are saved for comparison.