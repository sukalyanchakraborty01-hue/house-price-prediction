import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample dataset
data = {
    "area": [1000, 1500, 2000, 2500, 3000],
    "rooms": [2, 3, 3, 4, 5],
    "price": [200000, 300000, 400000, 500000, 600000]
}

df = pd.DataFrame(data)

X = df[["area", "rooms"]]
y = df["price"]

model = LinearRegression()
model.fit(X, y)

# Prediction
prediction = model.predict([[1800, 3]])

print("Predicted House Price:", prediction[0])
