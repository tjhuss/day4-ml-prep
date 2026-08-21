# ML-Ready Dataset Prep

Cleans two datasets into ML-ready form for classical regression practice.

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

## Usage

```
python3 day4_ml_prep.py
```

Re-running requires the raw `Datasets/houston_housing_2024.json` file to be
present locally (not included in this repo — see above).
