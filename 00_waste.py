import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.optimize import curve_fit

# 単位ごみ排出量・人口データ
df_waste = pd.read_csv("00_wastegen.csv")
df_pop = pd.read_csv("00_population.csv")

# 回帰式
def linear_func(t, a, b):
    return a + b * t

def linear_reg(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    if len(np.unique(x)) < 2:
        return float(y.mean()), 0.0
    popt, _ = curve_fit(linear_func, x, y)
    return popt[0], popt[1]

# 市町村(CITY)ごとの単位排出量(UNIT_WASTE)変化の算定
uw_params = {}

for city in df_waste["CITY_NAME"].dropna().unique():
    g = df_waste[df_waste["CITY_NAME"] == city].sort_values("YEAR")
    t = g["YEAR"].values
    y = g["UNIT_WASTE"].values

    if len(np.unique(t)) < 2:
        a_uw, b_uw = float(y.mean()), 0.0
    else:
        a_uw, b_uw = linear_reg(t, y)

    uw_params[city] = (a_uw, b_uw)

# 市町村ごとのごみ組成
fw_ratio    = df_waste.groupby("CITY_NAME")["FW_RATIO"].mean().fillna(0).to_dict()
paper_ratio = df_waste.groupby("CITY_NAME")["PAPER_RATIO"].mean().fillna(0).to_dict()
pla_ratio   = df_waste.groupby("CITY_NAME")["PLASTIC_RATIO"].mean().fillna(0).to_dict()
syntex_ratio = df_waste.groupby("CITY_NAME")["SYNTEX_RATIO"].mean().fillna(0).to_dict()
nattex_ratio = df_waste.groupby("CITY_NAME")["NATTEX_RATIO"].mean().fillna(0).to_dict()
wood_ratio   = df_waste.groupby("CITY_NAME")["WOOD_RATIO"].mean().fillna(0).to_dict()

# 農業集落ごとの人口の回帰
df_pop_rc = df_pop[(df_pop["CITY"] != 0) & (df_pop["RCOM"] != 0)]
group_cols = ["PREF", "CITY", "KCITY", "RCOM", "CITY_NAME"]

pop_params = {}
area_info = {}

for keys, g in df_pop_rc.groupby(group_cols):
    g = g.sort_values("YEAR")
    t = g["YEAR"].values
    y = g["POPULATION"].values

    if len(np.unique(t)) < 2:
        a_p, b_p = float(y.mean()), 0.0
    else:
        a_p, b_p = linear_reg(t, y)

    pop_params[keys] = (a_p, b_p)

    area_info[keys] = {
        "KEY": g["KEY"].iloc[0],
        "PREF": keys[0],
        "CITY": keys[1],
        "KCITY": keys[2],
        "RCOM": keys[3],
        "CITY_NAME": keys[4],
        "KCITY_NAME": g["KCITY_NAME"].iloc[0] if "KCITY_NAME" in g.columns else None,
    }

# 2030~2050年間年ごと値の算定
years = np.arange(2030, 2051)
rcom_rows = []

for keys, (a_p, b_p) in pop_params.items():
    pref, city_code, kcity, rcom, city_name = keys

    # 人口
    pop_future = linear_func(years, a_p, b_p)
    pop_future = np.where(pop_future < 0, 0, pop_future)
    pop_future = np.round(pop_future).astype(int)

    # 単位排出量→総排出量
    if city_name not in uw_params:
        a_uw, b_uw = 0.0, 0.0
        uw_future = np.zeros_like(years, dtype=float)
        fw = paper = pla = syntex = nattex = wood = 0.0
    else:
        a_uw, b_uw = uw_params[city_name]
        uw_future = linear_func(years, a_uw, b_uw)
        uw_future = np.where(uw_future < 0, 0, uw_future)

        fw     = fw_ratio.get(city_name, 0.0)
        paper  = paper_ratio.get(city_name, 0.0)
        pla    = pla_ratio.get(city_name, 0.0)
        syntex = syntex_ratio.get(city_name, 0.0)
        nattex = nattex_ratio.get(city_name, 0.0)
        wood   = wood_ratio.get(city_name, 0.0)

    total_waste = pop_future * uw_future
    food_waste = total_waste * fw
    pla_waste = total_waste * pla
    syntex_waste = total_waste * syntex

    total_waste = np.round(total_waste).astype(int)
    food_waste = np.round(food_waste).astype(int)
    pla_waste = np.round(pla_waste).astype(int)
    syntex_waste = np.round(syntex_waste).astype(int)

    info = area_info[keys]
    key = info["KEY"]
    kcity_name = info["KCITY_NAME"]

    for i, year in enumerate(years):
        rcom_rows.append({
            "KEY": key,
            "PREF": pref,
            "CITY": city_code,
            "KCITY": kcity,
            "RCOM": rcom,
            "CITY_NAME": city_name,
            "YEAR": year,
            "POPULATION": pop_future[i],
            "TOTAL_WASTE": total_waste[i],
            "FOOD_WASTE": food_waste[i],
            "PLA_WASTE": pla_waste[i],
            "SYNTEX_WASTE": syntex_waste[i],
        })

projection_rcom_df = pd.DataFrame(rcom_rows)

# 市町村排出量推移
city_base = (
    df_pop[(df_pop["CITY"] != 0) & (df_pop["KCITY"] == 0) & (df_pop["RCOM"] == 0)]
    .sort_values("YEAR")
    .groupby(["PREF", "CITY"])
    .first()
)

city_rows = []

for (pref, city_code), row in city_base.iterrows():
    city_key = row["KEY"]
    city_name = row["CITY_NAME"]
    kcity_name = row["KCITY_NAME"]  # ここは今は使っていないが残しておく

    for year in years:
        sub = projection_rcom_df[
            (projection_rcom_df["PREF"] == pref) &
            (projection_rcom_df["CITY"] == city_code) &
            (projection_rcom_df["YEAR"] == year)
        ]

        pop_sum = sub["POPULATION"].sum()
        tw_sum = sub["TOTAL_WASTE"].sum()
        fw_sum = sub["FOOD_WASTE"].sum()
        pla_sum = sub["PLA_WASTE"].sum()
        syntex_sum = sub["SYNTEX_WASTE"].sum()

        city_rows.append({
            "KEY": city_key,
            "PREF": pref,
            "CITY": city_code,
            "KCITY": 0,
            "RCOM": 0,
            "CITY_NAME": city_name,
            "YEAR": year,
            "POPULATION": int(pop_sum),
            "TOTAL_WASTE": tw_sum,
            "FOOD_WASTE": fw_sum,
            "PLA_WASTE": pla_sum,
            "SYNTEX_WASTE": syntex_sum,
        })

city_df = pd.DataFrame(city_rows)

# 京都府総量（都道府県レベル）
pref_base = (
    df_pop[(df_pop["CITY"] == 0) & (df_pop["KCITY"] == 0) & (df_pop["RCOM"] == 0)]
    .sort_values("YEAR")
    .groupby("PREF")
    .first()
)

pref_rows = []

for pref, row in pref_base.iterrows():
    pref_key = row["KEY"]
    pref_city_name = row["CITY_NAME"]
    pref_kcity_name = row["KCITY_NAME"]

    for year in years:
        sub = city_df[
            (city_df["PREF"] == pref) &
            (city_df["YEAR"] == year)
        ]

        pop_sum = sub["POPULATION"].sum()
        tw_sum = sub["TOTAL_WASTE"].sum()
        fw_sum = sub["FOOD_WASTE"].sum()
        pla_sum = sub["PLA_WASTE"].sum()
        syntex_sum = sub["SYNTEX_WASTE"].sum()

        pref_rows.append({
            "KEY": pref_key,
            "PREF": pref,
            "CITY": 0,
            "KCITY": 0,
            "RCOM": 0,
            "CITY_NAME": pref_city_name,
            "YEAR": year,
            "POPULATION": int(pop_sum),
            "TOTAL_WASTE": tw_sum,
            "FOOD_WASTE": fw_sum,
            "PLA_WASTE": pla_sum,
            "SYNTEX_WASTE": syntex_sum,
        })

pref_df = pd.DataFrame(pref_rows)

# 統合
waste_df = pd.concat(
    [projection_rcom_df, city_df, pref_df],
    ignore_index=True
)

waste_df.to_csv("01_waste.csv", index=False)

# GISデータ(GPKG)統合
shapefile_path = "00_rcom.shp"

try:
    gdf = gpd.read_file(shapefile_path)

    for col in ["PREF", "CITY", "KCITY", "RCOM"]:
        if col in gdf.columns:
            gdf[col] = gdf[col].fillna(0).astype(int)
        if col in waste_df.columns:
            waste_df[col] = waste_df[col].fillna(0).astype(int)

    join_keys = ["PREF", "CITY", "KCITY", "RCOM"]

    gdf_merged = gdf.merge(
        waste_df,
        on=join_keys,
        how="inner"
    )

    gdf_merged.to_file(
        "01_waste.gpkg",
        layer="waste",
        driver="GPKG"
    )

except Exception as e:
    print("Shapefile merge skipped or failed:", e)