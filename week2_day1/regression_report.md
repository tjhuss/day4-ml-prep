# Regression Report

Comparing four regression approaches — Linear, Ridge, Lasso, and
Polynomial (degree 2) — against each other on two different datasets, to
see how model choice actually affects results in practice rather than
just in theory.

Both use the same 80/20 train/test split (`random_state=42`) as the Day 5
baselines, so these numbers are directly comparable to that earlier work.

## Auto MPG (predicting mpg)

| Model | MAE | RMSE | R² |
| --- | --- | --- | --- |
| Linear | 2.46 | 3.26 | 0.7923 |
| Ridge | 2.46 | 3.25 | 0.7927 |
| Lasso | 2.53 | 3.26 | 0.7912 |
| Polynomial (degree 2) | **2.04** | **2.77** | **0.8500** |

Polynomial wins clearly here. That tracks with intuition — the way
horsepower and weight affect fuel efficiency probably isn't a flat,
additive relationship; a car that's both heavy *and* powerful likely
burns worse mileage than either trait alone would predict. Polynomial
features let the model capture that kind of interaction, and with only 8
features and 313 training rows, there's not enough complexity being added
to cause problems.

## Houston Housing (predicting price)

| Model | MAE | RMSE | R² |
| --- | --- | --- | --- |
| Linear | $102,454 | $160,521 | 0.7874 |
| Ridge | $101,038 | $157,339 | 0.7957 |
| Lasso | $101,380 | $157,826 | 0.7945 |
| Polynomial (degree 2) | $111,853 | $185,868 | 0.7149 |

Here the result flips — Polynomial is the *worst* performer, not the
best. With 12+ features (8 numeric plus several one-hot encoded home-type
columns), squaring and cross-multiplying everything explodes into 70-100+
columns, while there are only 320 rows to train on. That's overfitting:
the model has enough freedom to start fitting noise specific to the
training set rather than a real, generalizable pattern, so it looks
worse the moment it hits data it hasn't seen. Ridge and Lasso both edge
out plain Linear slightly here, which makes sense too — Houston's
features (beds, baths, area) are naturally correlated with each other,
and regularization is specifically suited to handling that.

## Takeaway

Same four models, same evaluation approach, opposite winner depending on
the dataset. There's no universally "best" regression model — it depends
on how many features you're working with relative to how much data you
have, and whether the real relationship is actually linear or not.
Polynomial regression isn't strictly "more powerful" than Linear; it's a
tradeoff that can just as easily backfire.
