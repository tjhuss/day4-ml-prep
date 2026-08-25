# ML-Ready Dataset Prep and Regression Models

Cleans two datasets into ML-ready form, trains a baseline regression
model on each, then compares four different regression approaches
(Linear, Ridge, Lasso, Polynomial) to see how model choice actually
plays out on real data.

## Setup

```
pip install -r requirements.txt
```

## Datasets

### Auto MPG
Predicts a car's fuel efficiency (mpg) from its specs. Sourced from the
[seaborn example datasets repo](https://github.com/mwaskom/seaborn-data).

- Started at 398 rows; 6 were missing `horsepower` and were dropped
- No duplicate rows; `origin` values were already consistent
- Output: `Datasets/auto_mpg_cleaned.csv` (392 rows)

### Houston Housing
Predicts a home's price from its specs (beds, baths, area, location, tax
value, etc.). Sourced from a Zillow-scraped Houston, TX listings export
(25,948 listings). The raw JSON file (143MB) is excluded from this repo via
`.gitignore` since it exceeds GitHub's file size limit — only the cleaned
output is committed.

Cleaning steps applied, in order:
1. Filtered to listings actually in Houston, TX (the source data included
   some nearby-city listings)
2. Dropped duplicate listings (by property ID)
3. Dropped listings missing any core field (price, beds, baths, area,
   zipcode, home type, tax value, lot size, lat/long, days on market)
4. Removed listings with an invalid ($0 or negative) price
5. Capped price to $2M to remove a small number of luxury-mansion outliers
   that would otherwise skew a regression model trained on mostly
   $75K-$800K homes
6. Removed a listing with 0 beds/0 baths despite being labeled
   "single family" (a data entry error, not a real 0-bedroom house)
7. Randomly sampled 400 rows from the remaining ~6,246 clean candidates
   (fixed random seed for reproducibility)

Output: `Datasets/houston_housing_cleaned.csv` (400 rows, no missing values,
no duplicates, price range $40K-$2M)

## Project structure

| Folder | Contents |
| --- | --- |
| `Datasets/` | Shared input/output CSVs, used by all three scripts below |
| `day4/` | `day4_ml_prep.py` — cleans both datasets |
| `day5/` | `day5_baseline_models.ipynb` — trains a baseline Linear Regression on each |
| `week2_day1/` | `week2_day1_regression_models.py` and `regression_report.md` — compares four regression approaches |

## Baseline models

`day5/day5_baseline_models.ipynb` trains a plain Linear Regression baseline for
each dataset (an 80/20 train/test split, no tuning or feature engineering)
to establish a reference point that future, more advanced models should
beat.

Notably, the Houston housing model deliberately excludes Zillow's own
`zestimate` field from its features — including it would be a data leakage
bug, since it's essentially another estimate of the exact value being
predicted.

| Dataset | MAE | RMSE | R² |
| --- | --- | --- | --- |
| Auto MPG (mpg) | 2.46 | 3.26 | 0.79 |
| Houston Housing (price) | $102,454 | $160,521 | 0.79 |

## Regression model comparison

`week2_day1/week2_day1_regression_models.py` trains Linear, Ridge, Lasso,
and Polynomial (degree 2) regression on both datasets. Full writeup with
explanations in [`week2_day1/regression_report.md`](week2_day1/regression_report.md)
— short version: Polynomial wins clearly on Auto MPG (R² 0.85 vs. ~0.79
for the others), but is the *worst* performer on Houston housing (R² 0.71
vs. ~0.79), because the larger housing feature set combined with
polynomial expansion overfits on only 320 training rows. There's no
single "best" model across both datasets.

## Usage

Each folder's script reads/writes to the shared `Datasets/` folder using
a relative path, so `cd` into the folder first:

```
cd day4
python3 day4_ml_prep.py

cd ../day5
jupyter nbconvert --to notebook --execute --inplace day5_baseline_models.ipynb
# or just open day5_baseline_models.ipynb in Jupyter/VS Code and run all cells

cd ../week2_day1
python3 week2_day1_regression_models.py
```

Re-running `day4_ml_prep.py` requires the raw `Datasets/houston_housing_2024.json`
file to be present locally (not included in this repo — see above).
