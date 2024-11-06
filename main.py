import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import tkinter as tk
from tkinter import ttk, messagebox
matplotlib.use('TkAgg')

def GoldForecasting():
    def plot_data():
        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(16, 20))
        sns.scatterplot(data=data, x='Crude_Oil', y='Gold_Price', ax=axes[0, 0])
        sns.scatterplot(data=data, x='Interest_Rate', y='Gold_Price', ax=axes[0, 1])
        sns.scatterplot(data=data, x='USD_INR', y='Gold_Price', ax=axes[1, 0])
        sns.scatterplot(data=data, x='Sensex', y='Gold_Price', ax=axes[1, 1])
        sns.scatterplot(data=data, x='CPI', y='Gold_Price', ax=axes[2, 0])
        sns.scatterplot(data=data, x='USD_Index', y='Gold_Price', ax=axes[2, 1])
        if 'Date' in data.columns:
            sns.lineplot(data=data, x='Date', y='Gold_Price', ax=axes[3, 0])
            axes[3, 0].set_title('Gold Price Over Time')
        else:
            axes[3, 0].remove()
        plt.tight_layout()
        plt.show()
    def plot_regression_model():
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=gold_y_test, y=gold_y_pred)
        plt.plot([0, max(gold_y_test)], [0, max(gold_y_test)], color='red', linestyle='--')
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('Actual vs Predicted Price')
        plt.show()
    def predict_gold_price():
        try:
            crude_oil = float(crude_oil_entry.get())
            interest_rate = float(interest_rate_entry.get())
            usd_inr = float(usd_inr_entry.get())
            sensex = float(sensex_entry.get())
            cpi = float(cpi_entry.get())
            usd_index = float(usd_index_entry.get())
            input_dict = {
                'Crude_Oil': crude_oil,
                'Interest_Rate': interest_rate,
                'USD_INR': usd_inr,
                'Sensex': sensex,
                'CPI': cpi,
                'USD_Index': usd_index
            }
            input_data = pd.DataFrame([input_dict])
            for col in gold_model_columns:
                if col not in input_data.columns:
                    input_data[col] = 0
            input_data = input_data[gold_model_columns]
            prediction = gold_model.predict(input_data)[0]
            messagebox.showinfo("Prediction", f"Predicted Gold Price: ${prediction:.2f}")
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid input values.")
    def show_mse():
        messagebox.showinfo("Mean Squared Error", f"Mean Squared Error: {mse_gold:.2f}")
    gold_data_url = ""
    gold_data = pd.read_csv(gold_data_url)
    data = gold_data
    if 'Date' in gold_data.columns:
        gold_data['Date'] = pd.to_datetime(gold_data['Date'])
    gold_data = gold_data[['Gold_Price', 'Crude_Oil', 'Interest_Rate', 'USD_INR', 'Sensex', 'CPI', 'USD_Index']]
    gold_data = gold_data.dropna()
    gold_X = gold_data.drop('Gold_Price', axis=1)
    gold_y = gold_data['Gold_Price']
    gold_X_train, gold_X_test, gold_y_train, gold_y_test = train_test_split(gold_X, gold_y, test_size=0.2, random_state=42)
    gold_model = LinearRegression()
    gold_model.fit(gold_X_train, gold_y_train)
    gold_y_pred = gold_model.predict(gold_X_test)
    mse_gold = mean_squared_error(gold_y_test, gold_y_pred)
    print(f"Mean Squared Error: {mse_gold}")
    gold_model_columns = list(gold_X.columns)
    root = tk.Tk()
    root.title("Linear Regression Prediction")
    tk.Label(root, text="Crude Oil:").grid(row=0, column=0)
    crude_oil_entry = tk.Entry(root)
    crude_oil_entry.grid(row=0, column=1)
    tk.Label(root, text="Interest Rate:").grid(row=1, column=0)
    interest_rate_entry = tk.Entry(root)
    interest_rate_entry.grid(row=1, column=1)
    tk.Label(root, text="USD/INR:").grid(row=2, column=0)
    usd_inr_entry = tk.Entry(root)
    usd_inr_entry.grid(row=2, column=1)
    tk.Label(root, text="Sensex:").grid(row=3, column=0)
    sensex_entry = tk.Entry(root)
    sensex_entry.grid(row=3, column=1)
    tk.Label(root, text="CPI:").grid(row=4, column=0)
    cpi_entry = tk.Entry(root)
    cpi_entry.grid(row=4, column=1)
    tk.Label(root, text="USD Index:").grid(row=5, column=0)
    usd_index_entry = tk.Entry(root)
    usd_index_entry.grid(row=5, column=1)
    tk.Button(root, text="Predict", command=predict_gold_price).grid(row=6, column=0, columnspan=2)
    tk.Button(root, text="Plot Data", command=plot_data).grid(row=7, column=0, columnspan=2)
    tk.Button(root, text="Plot Regression Model", command=plot_regression_model).grid(row=8, column=0, columnspan=2)
    tk.Button(root, text="Show MSE", command=show_mse).grid(row=9, column=0, columnspan=2)
    result_label = tk.Label(root, text="Predicted Charges: $0.00")
    result_label.grid(row=10, column=0, columnspan=2)
    root.mainloop()

def MedicalInsurance():
    def plot_data():
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 18))
        sns.scatterplot(data=data, x='age', y='charges', ax=axes[0, 0])
        sns.scatterplot(data=data, x='bmi', y='charges', ax=axes[0, 1])
        sns.scatterplot(data=data, x='children', y='charges', ax=axes[1, 0])
        sns.boxplot(data=data, x='sex', y='charges', ax=axes[1, 1])
        sns.boxplot(data=data, x='smoker', y='charges', ax=axes[2, 0])
        sns.boxplot(data=data, x='region', y='charges', ax=axes[2, 1])
        plt.tight_layout()
        plt.show()
    def predict_charges():
        try:
            age = float(age_entry.get())
            bmi = float(bmi_entry.get())
            children = int(children_entry.get())
            smoker = smoker_var.get()
            region = region_var.get()
            sex = sex_var.get()
            input_dict = {
                'age': age,
                'bmi': bmi,
                'children': children,
                'smoker_yes': 1 if smoker == 'yes' else 0,
                'region_northeast': 1 if region == 'northeast' else 0,
                'region_northwest': 1 if region == 'northwest' else 0,
                'region_southeast': 1 if region == 'southeast' else 0,
                'region_southwest': 1 if region == 'southwest' else 0,
                'sex_male': 1 if sex == 'male' else 0
            }
            input_data = pd.DataFrame([input_dict])
            for col in model_columns:
                if col not in input_data.columns:
                    input_data[col] = 0
            input_data = input_data[model_columns]
            prediction = model.predict(input_data)[0]
            messagebox.showinfo("Prediction", f"Predicted Charges: ${prediction:.2f}")
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid input values.")
    def plot_actual_vs_predicted():
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=y_pred)
        plt.plot([0, max(y_test)], [0, max(y_test)], color='red', linestyle='--')
        plt.xlabel('Actual Charges')
        plt.ylabel('Predicted Charges')
        plt.title('Actual vs Predicted Charges')
        plt.show()
    def show_mse():
        messagebox.showinfo("Mean Squared Error", f"Mean Squared Error: {mse:.2f}")
    mic_data_url = ''
    mic_data = pd.read_csv(mic_data_url)
    data = mic_data
    mic_data = pd.get_dummies(mic_data, drop_first=False)
    X = mic_data.drop('charges', axis=1)
    y = mic_data['charges']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    model_columns = list(X.columns)
    root = tk.Tk()
    root.title("Medical Insurance Charges Prediction")
    tk.Label(root, text="Age:").grid(row=0, column=0)
    age_entry = tk.Entry(root)
    age_entry.grid(row=0, column=1)
    tk.Label(root, text="BMI:").grid(row=1, column=0)
    bmi_entry = tk.Entry(root)
    bmi_entry.grid(row=1, column=1)
    tk.Label(root, text="Children:").grid(row=2, column=0)
    children_entry = tk.Entry(root)
    children_entry.grid(row=2, column=1)
    tk.Label(root, text="Smoker:").grid(row=3, column=0)
    smoker_var = tk.StringVar(value="no")
    tk.Radiobutton(root, text="Yes", variable=smoker_var, value="yes").grid(row=3, column=1)
    tk.Radiobutton(root, text="No", variable=smoker_var, value="no").grid(row=3, column=2)
    tk.Label(root, text="Region:").grid(row=4, column=0)
    region_var = tk.StringVar(value="northeast")
    tk.OptionMenu(root, region_var, "northeast", "northwest", "southeast", "southwest").grid(row=4, column=1)
    tk.Label(root, text="Sex:").grid(row=5, column=0)
    sex_var = tk.StringVar(value="male")
    tk.Radiobutton(root, text="Male", variable=sex_var, value="male").grid(row=5, column=1)
    tk.Radiobutton(root, text="Female", variable=sex_var, value="female").grid(row=5, column=2)
    tk.Button(root, text="Predict", command=predict_charges).grid(row=6, column=0, columnspan=2)
    tk.Button(root, text="Plot Data", command=plot_data).grid(row=7, column=0, columnspan=2)
    tk.Button(root, text="Plot Regression Model", command=plot_actual_vs_predicted).grid(row=8, column=0, columnspan=2)
    tk.Button(root, text="Show MSE", command=show_mse).grid(row=9, column=0, columnspan=2)
    result_label = tk.Label(root, text="Predicted Charges: $0.00")
    result_label.grid(row=10, column=0, columnspan=2)
    root.mainloop()

input1 = input("Press 1 for MEDICAL INSURANCE and Press 2 for GOLD PRICE\n")
if input1 == "1":
    MedicalInsurance()
elif input1 == "2":
    GoldForecasting()
else:
    print("Error: Invalid Input")
