import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load data
df = pd.read_csv("C:\\Users\\mayur\\Desktop\\house-price-project\\house_data.csv")
# Use only few features
X = df[['sqft_living', 'bedrooms', 'bathrooms']]
y = df['price']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open('model.pkl', 'wb'))

print("Model trained and saved")