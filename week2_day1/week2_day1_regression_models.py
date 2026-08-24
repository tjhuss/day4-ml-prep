import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mpg_df = pd.read_csv("../Datasets/auto_mpg_cleaned.csv")
mpg_df = pd.get_dummies(mpg_df, columns=['origin'], drop_first=True)
print(mpg_df.head())
print(mpg_df.columns.tolist())

feature_cols = ["cylinders", "displacement", "horsepower", "weight",
                 "acceleration", "model_year", "origin_japan", "origin_usa"]

X = mpg_df[feature_cols]
y = mpg_df["mpg"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train rows:", len(X_train))
print("Test rows:", len(X_test))

def evaluate(model, X_test, y_test, label):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    print(f"\n{label}")
    print(f"MAE:  {mae:.2f}")
    print(f"MSE:  {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.4f}")

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
evaluate(linear_model, X_test, y_test, "Linear Regression")

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
evaluate(ridge_model, X_test, y_test, "Ridge Regression")

lasso_model = Lasso(alpha=1.0)
lasso_model.fit(X_train, y_train)
evaluate(lasso_model, X_test, y_test, "Lasso Regression")   

poly_model = PolynomialFeatures(degree=2)
X_train_poly = poly_model.fit_transform(X_train)
X_test_poly = poly_model.transform(X_test)
poly_reg_model = LinearRegression()
poly_reg_model.fit(X_train_poly, y_train)
evaluate(poly_reg_model, X_test_poly, y_test, "Polynomial Regression (degree 2)")

##Model 2 for the the Houston Housing Dataset
housing_df = pd.read_csv("../Datasets/houston_housing_cleaned.csv")
housing_df = pd.get_dummies(housing_df, columns=["home_type"], drop_first=True)

housing_feature_cols = ["beds", "baths", "area", "latitude", "longitude",
                         "tax_assessed_value", "lot_area_value", "days_on_zillow"] + \
                        [col for col in housing_df.columns if col.startswith("home_type_")]

X2 = housing_df[housing_feature_cols]
y2 = housing_df["price"]

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)

print("\nHousing train rows:", len(X2_train))
print("Housing test rows:", len(X2_test))

linear_model2 = LinearRegression()
linear_model2.fit(X2_train, y2_train)
evaluate(linear_model2, X2_test, y2_test, "Linear Regression (Housing)")

ridge_model2 = Ridge(alpha=1.0)
ridge_model2.fit(X2_train, y2_train)
evaluate(ridge_model2, X2_test, y2_test, "Ridge Regression (Housing)")

lasso_model2 = Lasso(alpha=1.0)
lasso_model2.fit(X2_train, y2_train)
evaluate(lasso_model2, X2_test, y2_test, "Lasso Regression (Housing)")

poly2 = PolynomialFeatures(degree=2)
X2_train_poly = poly2.fit_transform(X2_train)
X2_test_poly = poly2.transform(X2_test)

poly_model2 = LinearRegression()
poly_model2.fit(X2_train_poly, y2_train)
evaluate(poly_model2, X2_test_poly, y2_test, "Polynomial Regression (Housing, degree 2)")
