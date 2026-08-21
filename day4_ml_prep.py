import pandas as pd 

df = pd.read_csv("Datasets/auto_mpg.csv")
print(df.head())
print(df.info())

df = df.dropna(subset=["horsepower"])  # Drop rows with missing values
print(df.info())
print("Duplicate rows:", df.duplicated().sum())
print(df["origin"].value_counts())

df.to_csv("Datasets/auto_mpg_cleaned.csv", index=False)
print("Saved auto_mpg_cleaned.csv with", len(df), "rows")


##Houston Housing Dataset Cleaning
import json

with open("Datasets/houston_housing_2024.json", "r") as f:
    listings = json.load(f)

print("Total listings:", len(listings))
print(listings[0].keys())  # Print the first listing to understand its structure

rows = []
for listing in listings:
    home_info = listing.get("hdpData", {}).get("homeInfo", {})
    lat_long = listing.get("latLong", {})

    rows.append({
        "zpid": listing.get("zpid"),
        "price": listing.get("unformattedPrice"),
        "street": listing.get("addressStreet"),
        "city": listing.get("addressCity"),
        "state": listing.get("addressState"),
        "zipcode": listing.get("addressZipcode"),
        "beds": listing.get("beds"),
        "baths": listing.get("baths"),
        "area": listing.get("area"),
        "latitude": lat_long.get("latitude"),
        "longitude": lat_long.get("longitude"),
        "home_type": home_info.get("homeType"),
        "tax_assessed_value": home_info.get("taxAssessedValue"),
        "lot_area_value": home_info.get("lotAreaValue"),
        "days_on_zillow": home_info.get("daysOnZillow"),
    })

df_housing = pd.DataFrame(rows)
print(df_housing.head())
print(df_housing.info())

# Keep only Houston, TX listings (a handful of nearby-city listings sometimes
# show up in Zillow search results)
df_housing = df_housing[(df_housing["city"] == "Houston") & (df_housing["state"] == "TX")]
print("\nAfter filtering to Houston, TX:", len(df_housing), "rows")

# Drop duplicate listings (same property id)
df_housing = df_housing.drop_duplicates(subset=["zpid"])
print("After dropping duplicate zpid:", len(df_housing), "rows")

# Drop rows missing any of our core fields
df_housing = df_housing.dropna(subset=[
    "price", "beds", "baths", "area", "zipcode",
    "home_type", "tax_assessed_value", "lot_area_value",
    "latitude", "longitude", "days_on_zillow",
])
print("After dropping missing values:", len(df_housing), "rows")

# Drop listings with a placeholder/invalid price
df_housing = df_housing[df_housing["price"] > 0]
print("After removing price <= 0:", len(df_housing), "rows")

# Cap to a reasonable price range so a handful of luxury outliers
# (e.g. multi-million dollar mansions) don't skew the regression dataset
df_housing = df_housing[df_housing["price"] <= 2_000_000]
print("After capping price to $2M:", len(df_housing), "rows")

# Drop listings with 0 beds/baths (likely data entry errors, not real homes)
df_housing = df_housing[(df_housing["beds"] > 0) & (df_housing["baths"] > 0)]
print("After removing 0 bed/bath listings:", len(df_housing), "rows")

print("\nHome type breakdown:")
print(df_housing["home_type"].value_counts())

# Sample down to our target size (random_state fixes the sample so it's
# reproducible if we re-run this script)
df_housing_sample = df_housing.sample(n=400, random_state=42)

df_housing_sample.to_csv("Datasets/houston_housing_cleaned.csv", index=False)
print(f"\nSaved houston_housing_cleaned.csv with {len(df_housing_sample)} rows")