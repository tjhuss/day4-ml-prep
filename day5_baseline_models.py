import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_baseline(df, target, feature_cols, categorical_cols, label):
    print(f"\n{'=' * 50}")
    print(f"Baseline model: {label}")
    print("=" * 50)

    X = df[feature_cols]
    y = df[target]

    # Linear Regression needs numeric input, so categorical columns get
    # one-hot encoded (turned into 0/1 columns, one per category).
    # drop_first=True drops one category per column to avoid redundancy.
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Hold back 20% of the data purely for testing. The model never sees
    # this during training, so it tells us how well it generalizes rather
    # than how well it memorized.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")
    print(f"MAE:  {mae:.2f}")
    print(f"MSE:  {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.4f}")

    coefficients = pd.Series(model.coef_, index=X.columns)
    coefficients = coefficients.reindex(coefficients.abs().sort_values(ascending=False).index)
    print("\nTop 5 most influential features:")
    print(coefficients.head(5))

    return model


mpg_df = pd.read_csv("Datasets/auto_mpg_cleaned.csv")
train_baseline(
    mpg_df,
    target="mpg",
    feature_cols=["cylinders", "displacement", "horsepower", "weight",
                  "acceleration", "model_year", "origin"],
    categorical_cols=["origin"],
    label="Auto MPG (predicting mpg)",
)

housing_df = pd.read_csv("Datasets/houston_housing_cleaned.csv")
train_baseline(
    housing_df,
    target="price",
    feature_cols=["beds", "baths", "area", "latitude", "longitude",
                  "tax_assessed_value", "lot_area_value", "days_on_zillow",
                  "home_type"],
    categorical_cols=["home_type"],
    label="Houston Housing (predicting price)",
)
