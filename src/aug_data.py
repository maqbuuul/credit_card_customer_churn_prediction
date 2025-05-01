import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)


def expand_and_augment_telecom_dataset(
    original_data_path="../data/raw/telecom_customer_churn.csv",
    zipcode_data_path="../data/raw/telecom_zipcode_population.csv",
    target_size=15000,
):
    """
    A complete solution to expand the telecom dataset to ~15,000 records
    and add new features that help predict customer churn, without introducing data leakage.
    """
    print("======= TELECOM DATASET EXPANSION AND AUGMENTATION =======")
    print(f"Loading original dataset from {original_data_path}")

    # Load original data
    df = pd.read_csv(original_data_path)
    zipcode_data = pd.read_csv(zipcode_data_path)

    original_size = len(df)
    print(f"Original dataset size: {original_size} records")

    # Calculate how many new records to generate
    # Add a small random variation to make it look more natural
    additional_records = target_size - original_size
    additional_records = int(additional_records * random.uniform(0.95, 1.05))
    print(f"Will generate {additional_records} new records")

    print("\n--- PART 1: DATASET EXPANSION ---")
    expanded_df = expand_dataset(df, additional_records)

    print(f"\nExpanded dataset created with {len(expanded_df)} records")

    print("\n--- PART 2: FEATURE AUGMENTATION ---")
    final_df = augment_dataset(expanded_df, zipcode_data)

    # Save final dataset
    final_filename = "../data/augmented_data/telecom_customer_churn_final.csv"
    final_df.to_csv(final_filename, index=False)

    print(f"\nFinal augmented dataset saved to {final_filename}")
    print(f"Total records: {len(final_df)}")
    print(f"Total features: {len(final_df.columns)}")

    # Generate a summary of what was added
    new_features = [col for col in final_df.columns if col not in df.columns]
    print(f"\nAdded {len(new_features)} new features:")
    for feature in new_features:
        print(f"- {feature}")

    print("\nProcess complete! Your dataset is ready for analysis.")
    return final_df


def expand_dataset(df, additional_records):
    """
    Expand the dataset by creating additional realistic customer records.
    """
    # Step 1: Create a new dataframe for the synthetic records
    new_records = []

    # Step 2: Generate new customer IDs (ensure they're different from existing ones)
    existing_ids = set(df["Customer ID"])

    # Helper function to generate a new customer ID
    def generate_customer_id():
        while True:
            # Format similar to existing IDs (4 digits + 5 letters)
            new_id = f"{random.randint(1000, 9999):04d}-{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=5))}"
            if new_id not in existing_ids:
                existing_ids.add(new_id)
                return new_id

    # Step 3: Create new records based on variations of existing customers
    print("Generating new customer records...")

    # Get all available zip codes
    all_zip_codes = df["Zip Code"].unique()

    # Get all cities and create city-zip mapping
    city_zip_mapping = (
        df[["City", "Zip Code"]]
        .drop_duplicates()
        .set_index("Zip Code")["City"]
        .to_dict()
    )

    # Get distribution of customer statuses to make it realistic
    status_distribution = df["Customer Status"].value_counts(normalize=True).to_dict()

    # Get available values for categorical fields
    available_offers = df["Offer"].unique()
    available_internet_types = [
        t for t in df["Internet Type"].unique() if isinstance(t, str) and pd.notna(t)
    ]
    available_contracts = df["Contract"].unique()
    available_payment_methods = df["Payment Method"].unique()

    # Get churn categories and reasons
    churn_mapping = df[df["Customer Status"] == "Churned"][
        ["Churn Category", "Churn Reason"]
    ].drop_duplicates()
    churn_combinations = list(
        zip(churn_mapping["Churn Category"], churn_mapping["Churn Reason"])
    )

    # Track progress
    progress_interval = max(1, additional_records // 10)

    for i in range(additional_records):
        if (i + 1) % progress_interval == 0:
            print(
                f"Generated {i + 1}/{additional_records} records ({(i + 1) / additional_records * 100:.1f}%)"
            )

        # Start with a random existing customer to use as base
        base_customer = df.iloc[random.randint(0, len(df) - 1)].copy()

        # Create new record with variations
        new_record = {}

        # Generate new ID
        new_record["Customer ID"] = generate_customer_id()

        # Demographic information (vary slightly from base)
        new_record["Gender"] = random.choice(["Male", "Female"])
        new_record["Age"] = max(
            18, min(90, int(base_customer["Age"]) + random.randint(-5, 5))
        )
        new_record["Married"] = random.choice(["Yes", "No"])
        new_record["Number of Dependents"] = max(
            0, min(6, base_customer["Number of Dependents"] + random.randint(-1, 1))
        )

        # Location - either use same or nearby zip code
        if random.random() < 0.7:  # 70% same zip code
            new_record["Zip Code"] = base_customer["Zip Code"]
            new_record["City"] = base_customer["City"]
            new_record["Latitude"] = base_customer["Latitude"] + random.uniform(
                -0.01, 0.01
            )
            new_record["Longitude"] = base_customer["Longitude"] + random.uniform(
                -0.01, 0.01
            )
        else:  # 30% different zip code
            new_zip = random.choice(all_zip_codes)
            new_record["Zip Code"] = new_zip
            new_record["City"] = city_zip_mapping.get(new_zip, "Unknown")

            # Find a customer with this zip code to get approximate coordinates
            zip_sample = df[df["Zip Code"] == new_zip]
            if len(zip_sample) > 0:
                sample_row = zip_sample.iloc[0]
                new_record["Latitude"] = sample_row["Latitude"] + random.uniform(
                    -0.01, 0.01
                )
                new_record["Longitude"] = sample_row["Longitude"] + random.uniform(
                    -0.01, 0.01
                )
            else:
                new_record["Latitude"] = base_customer["Latitude"] + random.uniform(
                    -0.5, 0.5
                )
                new_record["Longitude"] = base_customer["Longitude"] + random.uniform(
                    -0.5, 0.5
                )

        # Referrals - most have 0-2
        referral_weights = [
            0.6,
            0.25,
            0.1,
            0.03,
            0.02,
        ]  # Weights for 0,1,2,3,4 referrals
        new_record["Number of Referrals"] = np.random.choice(
            range(5), p=referral_weights
        )

        # Tenure - generate realistic distribution
        tenure_choices = [
            random.randint(1, 6),  # New customer (25% chance)
            random.randint(7, 24),  # Medium tenure (45% chance)
            random.randint(25, 72),  # Long tenure (30% chance)
        ]
        tenure_weights = [0.25, 0.45, 0.30]
        new_record["Tenure in Months"] = np.random.choice(
            tenure_choices, p=tenure_weights
        )

        # Offers - distribution similar to original
        new_record["Offer"] = random.choice(available_offers)

        # Services
        new_record["Phone Service"] = random.choice(["Yes", "No"])

        if new_record["Phone Service"] == "Yes":
            new_record["Avg Monthly Long Distance Charges"] = round(
                random.uniform(0, 60), 2
            )
            new_record["Multiple Lines"] = random.choice(["Yes", "No"])
        else:
            new_record["Avg Monthly Long Distance Charges"] = 0
            new_record["Multiple Lines"] = "No"

        # Internet service
        new_record["Internet Service"] = random.choice(["Yes", "No"])

        if new_record["Internet Service"] == "Yes":
            new_record["Internet Type"] = random.choice(available_internet_types)
            new_record["Avg Monthly GB Download"] = round(random.uniform(5, 350), 1)

            # More services for fiber customers
            has_feature_prob = (
                0.7 if new_record["Internet Type"] == "Fiber Optic" else 0.4
            )

            for feature in [
                "Online Security",
                "Online Backup",
                "Device Protection Plan",
                "Premium Tech Support",
                "Streaming TV",
                "Streaming Movies",
                "Streaming Music",
                "Unlimited Data",
            ]:
                new_record[feature] = (
                    "Yes" if random.random() < has_feature_prob else "No"
                )
        else:
            new_record["Internet Type"] = None
            new_record["Avg Monthly GB Download"] = 0
            for feature in [
                "Online Security",
                "Online Backup",
                "Device Protection Plan",
                "Premium Tech Support",
                "Streaming TV",
                "Streaming Movies",
                "Streaming Music",
                "Unlimited Data",
            ]:
                new_record[feature] = "No"

        # Contract and billing
        new_record["Contract"] = random.choice(available_contracts)
        new_record["Paperless Billing"] = random.choice(["Yes", "No"])
        new_record["Payment Method"] = random.choice(available_payment_methods)

        # Financial metrics
        # Base monthly charge on services
        service_count = sum(
            1
            for feature in [
                "Phone Service",
                "Internet Service",
                "Online Security",
                "Online Backup",
                "Device Protection Plan",
                "Premium Tech Support",
                "Streaming TV",
                "Streaming Movies",
                "Streaming Music",
            ]
            if new_record[feature] == "Yes"
        )

        # Calculate a realistic monthly charge
        base_charge = 0
        if new_record["Phone Service"] == "Yes":
            base_charge += random.uniform(25, 35)
            if new_record["Multiple Lines"] == "Yes":
                base_charge += random.uniform(10, 15)

        if new_record["Internet Service"] == "Yes":
            if new_record["Internet Type"] == "Fiber Optic":
                base_charge += random.uniform(65, 85)
            elif new_record["Internet Type"] == "Cable":
                base_charge += random.uniform(45, 60)
            else:  # DSL
                base_charge += random.uniform(30, 45)

            # Add costs for additional services
            add_on_services = [
                "Online Security",
                "Online Backup",
                "Device Protection Plan",
                "Premium Tech Support",
                "Streaming TV",
                "Streaming Movies",
                "Streaming Music",
            ]
            for service in add_on_services:
                if new_record[service] == "Yes":
                    base_charge += random.uniform(5, 10)

            if new_record["Unlimited Data"] == "Yes":
                base_charge += random.uniform(10, 20)

        # Adjust for contract discounts
        if new_record["Contract"] == "One Year":
            base_charge *= random.uniform(0.9, 0.95)  # 5-10% discount
        elif new_record["Contract"] == "Two Year":
            base_charge *= random.uniform(0.8, 0.9)  # 10-20% discount

        # Final monthly charge with some random variation
        new_record["Monthly Charge"] = round(
            base_charge * random.uniform(0.95, 1.05), 2
        )

        # Calculate total charges based on tenure
        new_record["Total Charges"] = round(
            new_record["Monthly Charge"]
            * new_record["Tenure in Months"]
            * random.uniform(0.95, 1.02),
            2,
        )  # Small variation for price changes

        # Other financial metrics
        new_record["Total Refunds"] = round(
            random.uniform(0, 50) if random.random() < 0.15 else 0, 2
        )
        new_record["Total Extra Data Charges"] = round(
            random.uniform(0, 150)
            if new_record["Unlimited Data"] == "No" and random.random() < 0.3
            else 0,
            2,
        )
        new_record["Total Long Distance Charges"] = round(
            new_record["Avg Monthly Long Distance Charges"]
            * new_record["Tenure in Months"],
            2,
        )

        # Total revenue
        new_record["Total Revenue"] = (
            new_record["Total Charges"]
            - new_record["Total Refunds"]
            + new_record["Total Extra Data Charges"]
            + new_record["Total Long Distance Charges"]
        )

        # Customer status and churn info (use realistic distribution)
        new_record["Customer Status"] = np.random.choice(
            list(status_distribution.keys()), p=list(status_distribution.values())
        )

        # Add churn reason for churned customers
        if new_record["Customer Status"] == "Churned":
            churn_category, churn_reason = random.choice(churn_combinations)
            new_record["Churn Category"] = churn_category
            new_record["Churn Reason"] = churn_reason
        else:
            new_record["Churn Category"] = ""
            new_record["Churn Reason"] = ""

        # Add to our collection
        new_records.append(new_record)

    # Create DataFrame from new records
    new_df = pd.DataFrame(new_records)

    # Combine with original data
    combined_df = pd.concat([df, new_df], ignore_index=True)
    return combined_df


def augment_dataset(df, zipcode_data):
    """
    Add new predictive features to the dataset that can help predict customer churn,
    WITHOUT introducing data leakage.
    """
    print("Adding new features to the dataset...")

    # Add support features (without leakage)
    df = add_support_features(df)
    print("✓ Added customer support features")

    # Add network quality features (without leakage)
    df = add_network_features(df)
    print("✓ Added network quality features")

    # Add usage pattern features (without leakage)
    df = add_usage_pattern_features(df)
    print("✓ Added usage pattern features")

    # Add competitive market features (without leakage)
    df = add_competitive_market_features(df, zipcode_data)
    print("✓ Added competitive market features")

    # Add customer engagement features (without leakage)
    df = add_customer_engagement_features(df)
    print("✓ Added customer engagement features")

    return df


# 1. CUSTOMER SUPPORT FEATURES
# =======================================================
def add_support_features(df):
    """
    Add customer support interaction features without using churn status
    """
    # Number of support tickets in last 6 months
    # Create realistic distribution based on service type and other factors, NOT churn status

    # Base tickets on internet type (Fiber has more issues than DSL)
    df["support_tickets_6m"] = np.zeros(len(df))
    fiber_users = df["Internet Type"] == "Fiber Optic"
    cable_users = df["Internet Type"] == "Cable"
    dsl_users = df["Internet Type"] == "DSL"
    no_internet = df["Internet Service"] == "No"

    # Assign tickets based on service type
    df.loc[fiber_users, "support_tickets_6m"] = np.random.poisson(
        lam=3.0, size=fiber_users.sum()
    )
    df.loc[cable_users, "support_tickets_6m"] = np.random.poisson(
        lam=2.2, size=cable_users.sum()
    )
    df.loc[dsl_users, "support_tickets_6m"] = np.random.poisson(
        lam=1.8, size=dsl_users.sum()
    )
    df.loc[no_internet, "support_tickets_6m"] = np.random.poisson(
        lam=0.8, size=no_internet.sum()
    )

    # Customers with more services tend to have more tickets
    service_count = (
        (df["Phone Service"] == "Yes").astype(int)
        + (df["Internet Service"] == "Yes").astype(int)
        + (df["Online Security"] == "Yes").astype(int)
        + (df["Online Backup"] == "Yes").astype(int)
        + (df["Device Protection Plan"] == "Yes").astype(int)
        + (df["Premium Tech Support"] == "Yes").astype(int)
        + (df["Streaming TV"] == "Yes").astype(int)
        + (df["Streaming Movies"] == "Yes").astype(int)
        + (df["Streaming Music"] == "Yes").astype(int)
    )

    # Add 0-2 tickets based on number of services
    service_adjustment = np.random.uniform(0, 0.3, len(df)) * service_count
    df["support_tickets_6m"] = df["support_tickets_6m"] + service_adjustment
    df["support_tickets_6m"] = df["support_tickets_6m"].round().astype(int)

    # Support ticket resolution time - based on area, not churn status
    df["avg_resolution_hours"] = np.zeros(len(df))

    # Urban areas might have faster resolution due to more staff
    urban_zipcode = df["Zip Code"].isin(
        [90001, 90011, 90026, 90031, 90037, 90043, 90044, 90047, 90059, 90061, 90744]
    )
    suburban_zipcode = ~urban_zipcode

    df.loc[urban_zipcode, "avg_resolution_hours"] = np.random.gamma(
        shape=2, scale=4, size=urban_zipcode.sum()
    )
    df.loc[suburban_zipcode, "avg_resolution_hours"] = np.random.gamma(
        shape=3, scale=5, size=suburban_zipcode.sum()
    )

    # Ticket severity distribution
    # Create function to generate severity distribution based on services, not churn
    def generate_severity(size):
        return np.random.choice(
            ["Low", "Medium", "High"],
            size=size,
            p=[0.5, 0.35, 0.15],  # Standard distribution regardless of churn
        )

    df["ticket_severity"] = generate_severity(len(df))

    # Customer satisfaction score (CSAT) - based on resolver time and severity, not on churn
    df["support_csat_score"] = 5 - (df["avg_resolution_hours"] / 20).clip(0, 4)

    # Adjust for ticket severity
    severity_adjustment = np.zeros(len(df))
    severity_adjustment[df["ticket_severity"] == "Medium"] = -0.5
    severity_adjustment[df["ticket_severity"] == "High"] = -1.0
    df["support_csat_score"] = df["support_csat_score"] + severity_adjustment

    # Add some randomness
    df["support_csat_score"] = df["support_csat_score"] + np.random.normal(
        0, 0.5, len(df)
    )
    df["support_csat_score"] = df["support_csat_score"].clip(1, 5).round()

    return df


# 2. NETWORK QUALITY FEATURES
# =======================================================
def add_network_features(df):
    """
    Add network quality features without using churn status
    """
    # Create masks for different segments to create realistic patterns
    fiber_users = df["Internet Type"] == "Fiber Optic"
    cable_users = df["Internet Type"] == "Cable"
    dsl_users = df["Internet Type"] == "DSL"

    # Signal strength based on technology type and geography
    df["signal_strength_pct"] = np.zeros(len(df))

    # Fiber has generally good signal but can vary
    df.loc[fiber_users, "signal_strength_pct"] = (
        np.random.beta(12, 3, size=fiber_users.sum()) * 100
    )

    # Cable has moderate signal quality
    df.loc[cable_users, "signal_strength_pct"] = (
        np.random.beta(8, 3, size=cable_users.sum()) * 100
    )

    # DSL has more variable signal
    df.loc[dsl_users, "signal_strength_pct"] = (
        np.random.beta(6, 3, size=dsl_users.sum()) * 100
    )

    # For users without internet, set to 0
    non_internet = ~(fiber_users | cable_users | dsl_users)
    df.loc[non_internet, "signal_strength_pct"] = 0

    # Adjust signal strength based on geography (using latitude as proxy for region)
    # This creates natural geographic patterns without using churn status
    north_region = df["Latitude"] > df["Latitude"].median()

    # Apply small regional adjustments to create natural patterns
    region_adjustment = np.where(
        north_region,
        np.random.uniform(0, 5, len(df)),  # North region bonus
        np.random.uniform(-5, 0, len(df)),
    )  # South region penalty
    df["signal_strength_pct"] = df["signal_strength_pct"] + region_adjustment
    df["signal_strength_pct"] = df["signal_strength_pct"].clip(0, 100)

    # Network outages - based on geography and tenure
    df["network_outages_6m"] = np.zeros(len(df))

    # Base outages on internet type
    df.loc[fiber_users, "network_outages_6m"] = np.random.poisson(
        lam=1.0, size=fiber_users.sum()
    )
    df.loc[cable_users, "network_outages_6m"] = np.random.poisson(
        lam=1.8, size=cable_users.sum()
    )
    df.loc[dsl_users, "network_outages_6m"] = np.random.poisson(
        lam=2.2, size=dsl_users.sum()
    )

    # Adjust by urban/rural setting using population data
    high_population = df["Zip Code"].isin(
        [90001, 90011, 90026, 90031]
    )  # Example high population zip codes
    df.loc[high_population, "network_outages_6m"] = (
        df.loc[high_population, "network_outages_6m"] + 1
    )

    # Data speed as percentage of advertised speed based on technology
    df["actual_speed_vs_expected_pct"] = np.zeros(len(df))

    # Different technologies have different real-world performance characteristics
    df.loc[fiber_users, "actual_speed_vs_expected_pct"] = np.random.normal(
        85, 10, size=fiber_users.sum()
    )
    df.loc[cable_users, "actual_speed_vs_expected_pct"] = np.random.normal(
        75, 15, size=cable_users.sum()
    )
    df.loc[dsl_users, "actual_speed_vs_expected_pct"] = np.random.normal(
        65, 20, size=dsl_users.sum()
    )

    # Create a "peak hour" variable instead of using churn status
    peak_hour_impact = np.random.choice([0, 1], size=len(df), p=[0.3, 0.7])
    df.loc[peak_hour_impact == 1, "actual_speed_vs_expected_pct"] = (
        df.loc[peak_hour_impact == 1, "actual_speed_vs_expected_pct"] * 0.9
    )

    # Clamp speed percentages to realistic values
    df["actual_speed_vs_expected_pct"] = df["actual_speed_vs_expected_pct"].clip(
        10, 110
    )

    return df


# 3. USAGE PATTERN FEATURES
# =======================================================
def add_usage_pattern_features(df):
    """
    Add usage pattern features without using churn status
    """
    # Base the patterns on existing GB download data but add natural variations
    base_usage = df["Avg Monthly GB Download"].copy()

    # Create 6 months of historical usage data
    months = 6

    # Four common usage patterns (without using churn information)
    pattern_types = np.random.choice(
        ["stable", "increasing", "decreasing", "volatile"],
        size=len(df),
        p=[0.5, 0.2, 0.2, 0.1],
    )

    for i in range(months):
        month_name = f"data_usage_month_{months - i}"

        if i == 0:  # Current month
            df[month_name] = base_usage
        else:
            # Apply different natural patterns
            trend = np.ones(len(df))

            # Apply the patterns based on the pattern type, not churn status
            trend[pattern_types == "stable"] = 1.0 + np.random.normal(
                0, 0.05, (pattern_types == "stable").sum()
            )
            trend[pattern_types == "increasing"] = (
                1.0
                + (0.05 * i)
                + np.random.normal(0, 0.03, (pattern_types == "increasing").sum())
            )
            trend[pattern_types == "decreasing"] = (
                1.0
                - (0.05 * i)
                + np.random.normal(0, 0.03, (pattern_types == "decreasing").sum())
            )

            # Volatile pattern has larger random swings
            if (pattern_types == "volatile").sum() > 0:
                trend[pattern_types == "volatile"] = 1.0 + np.random.normal(
                    0, 0.15, (pattern_types == "volatile").sum()
                )

            # Apply the pattern
            df[month_name] = base_usage * trend

            # Ensure values are not negative
            df[month_name] = df[month_name].clip(lower=0)

    # Create derived metrics showing usage trends
    df["usage_trend_pct"] = (
        (df["data_usage_month_1"] / df["data_usage_month_6"]).replace(
            [np.inf, -np.inf, np.nan], 1
        )
        - 1
    ) * 100

    # Volatility in usage (standard deviation across months)
    usage_cols = [f"data_usage_month_{i + 1}" for i in range(months)]
    df["usage_volatility"] = df[usage_cols].std(axis=1)

    return df


# 4. COMPETITIVE MARKET FEATURES
# =======================================================
def add_competitive_market_features(df, zipcode_data):
    """
    Add market competition features without using churn status
    """
    # Merge population data
    df = pd.merge(df, zipcode_data, left_on="Zip Code", right_on="Zip Code", how="left")

    # Fill missing population data with median
    median_population = zipcode_data["Population"].median()
    df["Population"] = df["Population"].fillna(median_population)

    # Create market competition index (number of competitors) based on population
    zip_competitors = {}

    for zip_code in df["Zip Code"].unique():
        population = df.loc[df["Zip Code"] == zip_code, "Population"].iloc[0]

        # More populated areas tend to have more competition
        if population < 10000:
            competitors = np.random.choice([1, 2], p=[0.7, 0.3])
        elif population < 50000:
            competitors = np.random.choice([2, 3, 4], p=[0.3, 0.5, 0.2])
        else:
            competitors = np.random.choice([3, 4, 5], p=[0.2, 0.5, 0.3])

        zip_competitors[zip_code] = competitors

    # Add competition information to dataframe
    df["market_competitors"] = df["Zip Code"].map(zip_competitors)

    # Create price competitiveness score (how our price compares to market average)
    # For each zip code, create a reference "market price" for similar service
    zip_market_price = {}

    for zip_code in df["Zip Code"].unique():
        # Higher competition areas tend to have lower market prices
        competitors = zip_competitors[zip_code]
        population = df.loc[df["Zip Code"] == zip_code, "Population"].iloc[0]

        # Base market price factor (higher = more expensive market)
        if population > 100000:  # Urban areas
            base_factor = random.uniform(0.9, 1.1)
        elif population > 30000:  # Suburban areas
            base_factor = random.uniform(0.95, 1.15)
        else:  # Rural areas
            base_factor = random.uniform(1.0, 1.2)

        # Adjust for competition (more competitors = lower prices)
        competition_factor = 1 - (competitors * 0.03)

        zip_market_price[zip_code] = base_factor * competition_factor

    # Calculate relative price position (1.0 = market price, >1.0 = more expensive)
    df["price_vs_market"] = df.apply(
        lambda row: row["Monthly Charge"]
        / (
            zip_market_price[row["Zip Code"]]
            * (
                60
                if row["Internet Type"] == "Fiber Optic"
                else 45
                if row["Internet Type"] == "Cable"
                else 35
            )
        ),
        axis=1,
    )

    # Contract end proximity (0-1 scale, 1 = very close to end)
    df["contract_end_proximity"] = 0

    # For month-to-month, always near "end"
    month_to_month = df["Contract"] == "Month-to-Month"
    df.loc[month_to_month, "contract_end_proximity"] = np.random.uniform(
        0.7, 1.0, size=month_to_month.sum()
    )

    # For annual contracts, depends on tenure relative to 12/24 months
    one_year = df["Contract"] == "One Year"
    two_year = df["Contract"] == "Two Year"

    df.loc[one_year, "contract_end_proximity"] = (
        df.loc[one_year, "Tenure in Months"] % 12
    ) / 12
    df.loc[two_year, "contract_end_proximity"] = (
        df.loc[two_year, "Tenure in Months"] % 24
    ) / 24

    # Adjust to make higher values mean closer to end
    df["contract_end_proximity"] = 1 - df["contract_end_proximity"]

    # Competitor offer exposure (influenced by competition and contract timing)
    df["competitor_offer_exposure"] = (
        df["market_competitors"] * df["contract_end_proximity"] / 5
    )
    df["competitor_offer_exposure"] = df["competitor_offer_exposure"].clip(0, 1)

    return df


# 5. CUSTOMER ENGAGEMENT FEATURES
# =======================================================
def add_customer_engagement_features(df):
    """
    Add customer engagement metrics without using churn status
    """
    # Last bill payment behavior - based on contract type, not churn status
    # 0 = on time, 1 = 1-15 days late, 2 = 16-30 days late, 3 = 30+ days late
    df["last_payment_delay_category"] = np.zeros(len(df))

    # Different payment patterns based on contract type (not churn)
    month_to_month = df["Contract"] == "Month-to-Month"
    one_year = df["Contract"] == "One Year"
    two_year = df["Contract"] == "Two Year"

    # Month-to-month customers tend to have more variable payment timing
    df.loc[month_to_month, "last_payment_delay_category"] = np.random.choice(
        [0, 1, 2, 3], size=month_to_month.sum(), p=[0.7, 0.15, 0.1, 0.05]
    )

    # One year contract customers are more reliable
    df.loc[one_year, "last_payment_delay_category"] = np.random.choice(
        [0, 1, 2, 3], size=one_year.sum(), p=[0.8, 0.12, 0.06, 0.02]
    )

    # Two year contract customers are most reliable
    df.loc[two_year, "last_payment_delay_category"] = np.random.choice(
        [0, 1, 2, 3], size=two_year.sum(), p=[0.9, 0.07, 0.02, 0.01]
    )

    # Number of times logged into self-service portal - based on service complexity, not churn
    df["portal_logins_3m"] = np.zeros(len(df))

    # Calculate service complexity score
    service_count = (
        (df["Phone Service"] == "Yes").astype(int)
        + (df["Internet Service"] == "Yes").astype(int)
        + (df["Online Security"] == "Yes").astype(int)
        + (df["Online Backup"] == "Yes").astype(int)
        + (df["Device Protection Plan"] == "Yes").astype(int)
        + (df["Premium Tech Support"] == "Yes").astype(int)
        + (df["Streaming TV"] == "Yes").astype(int)
        + (df["Streaming Movies"] == "Yes").astype(int)
        + (df["Streaming Music"] == "Yes").astype(int)
        + (df["Unlimited Data"] == "Yes").astype(int)
    )

    # Generate logins based on service complexity
    low_complexity = service_count <= 3
    medium_complexity = (service_count > 3) & (service_count <= 6)
    high_complexity = service_count > 6

    df.loc[low_complexity, "portal_logins_3m"] = np.random.choice(
        [0, 1, 2, 3, 4], size=low_complexity.sum(), p=[0.3, 0.3, 0.2, 0.15, 0.05]
    )

    df.loc[medium_complexity, "portal_logins_3m"] = np.random.choice(
        [0, 1, 2, 3, 4, 5, 6],
        size=medium_complexity.sum(),
        p=[0.15, 0.2, 0.25, 0.2, 0.1, 0.05, 0.05],
    )

    df.loc[high_complexity, "portal_logins_3m"] = np.random.choice(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        size=high_complexity.sum(),
        p=[0.1, 0.1, 0.15, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05, 0.03, 0.02],
    )

    # Service changes requested - based on tenure and age, not churn status
    df["service_changes_6m"] = np.zeros(len(df))

    # New customers tend to adjust services more
    new_customers = df["Tenure in Months"] <= 6
    mid_tenure = (df["Tenure in Months"] > 6) & (df["Tenure in Months"] <= 24)
    long_tenure = df["Tenure in Months"] > 24

    # Age can influence technology adoption
    younger = df["Age"] < 35
    middle_aged = (df["Age"] >= 35) & (df["Age"] < 60)
    older = df["Age"] >= 60

    # Combine factors to estimate service change patterns (without using churn)
    # Younger new customers change services more frequently
    df.loc[new_customers & younger, "service_changes_6m"] = np.random.choice(
        [-1, 0, 1],  # -1 = downgrade, 0 = no change, 1 = upgrade
        size=(new_customers & younger).sum(),
        p=[0.2, 0.3, 0.5],  # More likely to upgrade
    )

    # Older long-tenure customers rarely change
    df.loc[long_tenure & older, "service_changes_6m"] = np.random.choice(
        [-1, 0, 1],
        size=(long_tenure & older).sum(),
        p=[0.15, 0.75, 0.1],  # Mostly no change
    )

    # Middle groups have balanced patterns
    df.loc[
        ~((new_customers & younger) | (long_tenure & older)), "service_changes_6m"
    ] = np.random.choice(
        [-1, 0, 1],
        size=(~((new_customers & younger) | (long_tenure & older))).sum(),
        p=[0.2, 0.6, 0.2],  # Balanced
    )

    return df


# Run the complete process if executed as a script
if __name__ == "__main__":
    expand_and_augment_telecom_dataset()
