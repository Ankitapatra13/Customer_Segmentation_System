import pandas as pd

def preprocess_data(df):
    df["Income"] = df["Income"].fillna(df["Income"].median())

    df["Age"] = 2026 - df["Year_Birth"]

    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
    ref_date = df["Dt_Customer"].max()
    df["CustomerTenureDays"] = (ref_date - df["Dt_Customer"]).dt.days

    df["TotalSpending"] = (
        df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] +
        df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
    )

    df["TotalChildren"] = df["Kidhome"] + df["Teenhome"]

    df["Education"] = df["Education"].replace({
        "Basic": "Undergraduate",
        "2n Cycle": "Undergraduate",
        "Graduation": "Graduate",
        "PhD": "Postgraduate",
        "Master": "Postgraduate"
    })

    df["LivingWith"] = df["Marital_Status"].replace({
        "Married": "Partner",
        "Together": "Partner",
        "Single": "Alone",
        "Divorced": "Alone",
        "Widow": "Alone",
        "Alone": "Alone"
    })

    drop_cols = [
        "ID","Year_Birth","Marital_Status","Kidhome","Teenhome","Dt_Customer",
        "MntWines","MntFruits","MntMeatProducts","MntFishProducts",
        "MntSweetProducts","MntGoldProds"
    ]

    df = df.drop(columns=drop_cols)

    df = df[(df["Age"] < 90) & (df["Income"] < 600000)]

    return df