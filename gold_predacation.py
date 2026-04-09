# ==========================================
# GOLD PRICE PREDICTION USING ML
# User enters date → predicts gold rate
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ------------------------------------------
# 1. LOAD DATA
# ------------------------------------------
gr = pd.read_csv(r'D:\prackets\org_gold.csv')

print("\nDataset Preview:\n")
print(gr.head())

# ------------------------------------------
# 2. PREPROCESSING
# ------------------------------------------
gr['Date'] = pd.to_datetime(gr['Date'])
gr = gr.sort_values('Date')

# create time index
gr['day_index'] = np.arange(len(gr))

# ------------------------------------------
# 3. VISUALIZATION
# ------------------------------------------
sns.regplot(x='USDIND', y="Gold Price(24 Karat)", data=gr)
plt.title("USDINR vs Gold Price")
plt.show()

# ------------------------------------------
# 4. TRAIN GOLD MODEL (USDINR -> GOLD)
# ------------------------------------------
from sklearn.model_selection import train_test_split

x = gr[['USDIND']]
y = gr[['Gold Price(24 Karat)']]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.40, random_state=42
)

# scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train_scaled, y_train)

# equation
m = regressor.coef_[0][0]
b = regressor.intercept_[0]

print("\nModel Equation:")
print("Gold =", m, "* USDINR +", b)

# ------------------------------------------
# 5. MODEL ACCURACY
# ------------------------------------------
from sklearn.metrics import r2_score, mean_squared_error

y_pred = regressor.predict(x_test_scaled)

print("\nR2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# ------------------------------------------
# 6. TRAIN USDINR FORECAST MODEL
# ------------------------------------------
usd_model = LinearRegression()
usd_model.fit(gr[['day_index']], gr['USDIND'])

# ------------------------------------------
# 7. USER INPUT DATE PREDICTION
# ------------------------------------------
print("\nEnter a future date to predict gold price")
print("Format: YYYY-MM-DD")

user_date = input("Enter date: ")

# convert to date
user_date = datetime.strptime(user_date, "%Y-%m-%d")

last_date = gr['Date'].max()
days_ahead = (user_date - last_date).days

if days_ahead <= 0:
    print("\n Date must be after", last_date.date())

else:
    # predict USDINR for that future day
    future_index = np.array([[gr['day_index'].max() + days_ahead]])
    predicted_usdinr = usd_model.predict(future_index)

    # convert USDINR -> GOLD
    scaled_val = scaler.transform(predicted_usdinr.reshape(-1,1))
    predicted_gold = regressor.predict(scaled_val)

    print("\n========== PREDICTION RESULT ==========")
    print("Date:", user_date.date())
    print("Predicted USDINR:", round(predicted_usdinr[0],2))
    print("Predicted Gold Price:", round(predicted_gold[0][0],2))
