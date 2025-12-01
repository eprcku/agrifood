import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.optimize import curve_fit

# 作付面積・農業集落比率・単位農業投入量データ
df_cropland = pd.read_csv("00_cropland.csv")
df_rcom = pd.read_csv("00_rcom.csv")
df_agroinput = pd.read_csv("00_agroinput.csv")

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

years_future = np.arange(2029, 2051)

# 面積の推定
df_crop_clean = df_cropland.dropna(subset=["YEAR", "EXTENT"]).copy()
df_crop_clean["YEAR"] = df_crop_clean["YEAR"].astype(float)
df_crop_clean["EXTENT"] = df_crop_clean["EXTENT"].astype(float)
df_crop_clean["CULTIVAR"] = df_crop_clean["CULTIVAR"].fillna("unknown")

if "CROP_NORM" not in df_crop_clean.columns:
    df_crop_clean["CROP_NORM"] = df_crop_clean["CROP"]

def fit_and_project(g):
    g = g.sort_values("YEAR")
    t = g["YEAR"].values
    y = g["EXTENT"].values
    a, b = linear_reg(t, y)
    y_future = a + b * years_future
    y_future = np.where(y_future < 0, 0, y_future)
    y_future = np.round(y_future).astype(int)
    return years_future, y_future

df_ukyo = df_crop_clean[df_crop_clean["CITY_NAME"] == "Ukyo"].copy()
df_other = df_crop_clean[df_crop_clean["CITY_NAME"] != "Ukyo"].copy()

results = []

# 右京区以外の市町村(CITY)
group_cols_other = ["PREF", "CITY", "CITY_NAME", "CROP", "CULTIVAR", "CROP_NORM"]
for keys, g in df_other.groupby(group_cols_other):
    pref, city_code, city_name, crop, cultivar, crop_norm = keys
    years, y_future = fit_and_project(g)
    for yr, ext in zip(years, y_future):
        results.append({
            "PREF": pref,
            "CITY": city_code,
            "KCITY": 0,
            "RCOM": 0,
            "CITY_NAME": city_name,
            "KCITY_NAME": None,
            "CROP": crop,
            "CULTIVAR": cultivar,
            "CROP_NORM": crop_norm,
            "YEAR": yr,
            "EXTENT_PROJECTED": ext
        })

# 右京区(京北地域統計問題)
group_cols_ukyo = ["PREF", "CITY", "KCITY", "CITY_NAME", "KCITY_NAME", "CROP", "CULTIVAR", "CROP_NORM"]
for keys, g in df_ukyo.groupby(group_cols_ukyo):
    pref, city_code, kcity, city_name, kcity_name, crop, cultivar, crop_norm = keys
    years, y_future = fit_and_project(g)
    for yr, ext in zip(years, y_future):
        results.append({
            "PREF": pref,
            "CITY": city_code,
            "KCITY": kcity,
            "RCOM": 0,
            "CITY_NAME": city_name,
            "KCITY_NAME": kcity_name,
            "CROP": crop,
            "CULTIVAR": cultivar,
            "CROP_NORM": crop_norm,
            "YEAR": yr,
            "EXTENT_PROJECTED": ext
        })

cropland_proj_df = pd.DataFrame(results)

# 農業集落(RCOM)ごとの面積配分
df_rcom_rc = df_rcom[df_rcom["RCOM"] != 0].copy()
df_rcom_city = df_rcom[(df_rcom["CITY"] != 0) & (df_rcom["RCOM"] == 0)].copy()
df_rcom_pref = df_rcom[(df_rcom["CITY"] == 0) & (df_rcom["RCOM"] == 0)].copy()

df_source = cropland_proj_df[
    (cropland_proj_df["CITY"] != 0) &
    (cropland_proj_df["RCOM"] == 0)
].copy()

df_keep = cropland_proj_df[
    ~(
        (cropland_proj_df["CITY"] != 0) &
        (cropland_proj_df["RCOM"] == 0)
    )
].copy()

def choose_ratio_col(crop_norm: str):
    if isinstance(crop_norm, str):
        cn = crop_norm.lower()
        if cn == "rice":
            return "PADDY_RATIO"
        if cn == "tea":
            return "TEA_RATIO"
        if cn == "veg":
            return "VEG_RATIO"
    return None

distributed_rows = []
consistency_records = []

for _, row in df_source.iterrows():
    pref = row["PREF"]
    city_code = row["CITY"]
    kcity = row["KCITY"]
    crop = row["CROP"]
    cultivar = row["CULTIVAR"]
    crop_norm = row.get("CROP_NORM", np.nan)
    year = row["YEAR"]
    total_extent = int(row["EXTENT_PROJECTED"])

    mask = (df_rcom_rc["PREF"] == pref) & (df_rcom_rc["CITY"] == city_code)
    if pd.notna(kcity) and int(kcity) != 0:
        mask &= (df_rcom_rc["KCITY"] == kcity)

    base_cols = ["KEY", "PREF", "CITY", "KCITY", "RCOM", "CITY_NAME", "KCITY_NAME"]

    ratio_col = choose_ratio_col(crop_norm)

    if ratio_col and (ratio_col in df_rcom_rc.columns):
        sub = df_rcom_rc.loc[mask, base_cols + [ratio_col]].copy()
        sub_valid = sub[sub[ratio_col].notna() & (sub[ratio_col] > 0)]
        if sub_valid.empty:
            sub = df_rcom_rc.loc[mask, base_cols].copy()
            use_equal = True
        else:
            sub = sub_valid
            use_equal = False
    else:
        sub = df_rcom_rc.loc[mask, base_cols].copy()
        use_equal = True

    if sub.empty:
        consistency_records.append({
            "PREF": pref,
            "CITY": city_code,
            "KCITY": kcity,
            "CROP": crop,
            "CULTIVAR": cultivar,
            "CROP_NORM": crop_norm,
            "YEAR": year,
            "TOTAL_ORIG": total_extent,
            "TOTAL_DISTRIBUTED": 0,
            "DIFF": total_extent
        })
        continue

    if use_equal:
        n = len(sub)
        ratios = np.ones(n, dtype=float) / float(n)
    else:
        vals = sub[ratio_col].astype(float).values
        total = vals.sum()
        if total <= 0:
            n = len(sub)
            ratios = np.ones(n, dtype=float) / float(n)
        else:
            ratios = vals / total

    raw_alloc = total_extent * ratios
    ints = np.floor(raw_alloc).astype(int)
    diff = total_extent - ints.sum()

    frac = raw_alloc - ints
    order = np.argsort(-frac)
    for i in range(diff):
        ints[order[i]] += 1

    distributed_sum = int(ints.sum())
    consistency_records.append({
        "PREF": pref,
        "CITY": city_code,
        "KCITY": kcity,
        "CROP": crop,
        "CULTIVAR": cultivar,
        "CROP_NORM": crop_norm,
        "YEAR": year,
        "TOTAL_ORIG": total_extent,
        "TOTAL_DISTRIBUTED": distributed_sum,
        "DIFF": total_extent - distributed_sum
    })

    for j, (_, r) in enumerate(sub.iterrows()):
        r_extent = max(0, int(ints[j]))
        distributed_rows.append({
            "KEY": r["KEY"],
            "PREF": r["PREF"],
            "CITY": r["CITY"],
            "KCITY": r["KCITY"],
            "RCOM": r["RCOM"],
            "CITY_NAME": r["CITY_NAME"],
            "KCITY_NAME": r["KCITY_NAME"],
            "CROP": crop,
            "CULTIVAR": cultivar,
            "CROP_NORM": crop_norm,
            "YEAR": year,
            "EXTENT_PROJECTED": r_extent
        })

consistency_df = pd.DataFrame(consistency_records)
print("Distribution groups:", len(consistency_df))
if not consistency_df.empty:
    print("Max abs DIFF:", consistency_df["DIFF"].abs().max())

df_distributed = pd.DataFrame(distributed_rows)

cropland_rcom_df = pd.concat(
    [df_distributed, df_keep[df_keep["RCOM"] != 0]],
    ignore_index=True
)

# 単位投入量の推定
df_crop_rc = cropland_rcom_df[
    (cropland_rcom_df["RCOM"] != 0) &
    (cropland_rcom_df["YEAR"].between(2029, 2050))
].copy()
df_crop_rc["CITY"] = pd.to_numeric(df_crop_rc["CITY"], errors="coerce")

unit_rows = []
df_rice = df_agroinput[df_agroinput["CROP"] == "rice"].copy()
df_tea = df_agroinput[df_agroinput["CROP"] == "tea"].copy()
df_veg = df_agroinput[df_agroinput["CROP"] == "veg"].copy()

# 水稲
for (crop, cultivar), g in df_rice.groupby(["CROP", "CULTIVAR"]):
    g = g.sort_values("YEAR")
    a, b = linear_reg(g["YEAR"], g["SYN"])
    syn_2020 = a + b * 2020.0
    syn_2050 = 0.7 * syn_2020

    def syn_func(t):
        return syn_2020 + (syn_2050 - syn_2020) * ((t - 2020.0) / 30.0)

    n_req = g["N_REQ"].dropna().iloc[0] if g["N_REQ"].notna().any() else 0.0
    g_last = g.iloc[-1]
    others = {col: (0.0 if pd.isna(g_last[col]) else g_last[col])
              for col in ["AGROCHEM", "DIESEL", "GASOL", "KEROS", "ENGOIL", "ELECT"]}

    for yr in years_future:
        syn = max(0.0, float(syn_func(yr)))
        org = max(0.0, n_req - syn)
        row = {
            "CITY": np.nan,
            "CROP": crop,
            "CULTIVAR": cultivar,
            "YEAR": yr,
            "SYN": syn,
            "ORG": org,
        }
        row.update(others)
        unit_rows.append(row)

# 茶
for (city, crop, cultivar), g in df_tea.groupby(["CITY", "CROP", "CULTIVAR"]):
    g = g.sort_values("YEAR")
    if 2020 not in g["YEAR"].values or 2050 not in g["YEAR"].values:
        continue

    syn_2020 = g.loc[g["YEAR"] == 2020, "SYN"].iloc[0]
    syn_2050 = g.loc[g["YEAR"] == 2050, "SYN"].iloc[0]

    def syn_func(t):
        return syn_2020 + (syn_2050 - syn_2020) * ((t - 2020.0) / 30.0)

    n_req = g["N_REQ"].dropna().iloc[0] if g["N_REQ"].notna().any() else 0.0
    g_2020 = g[g["YEAR"] == 2020].iloc[0]
    others = {col: (0.0 if pd.isna(g_2020[col]) else g_2020[col])
              for col in ["AGROCHEM", "DIESEL", "GASOL", "KEROS", "ENGOIL", "ELECT"]}

    for yr in years_future:
        syn = max(0.0, float(syn_func(yr)))
        org = max(0.0, n_req - syn)
        row = {
            "CITY": city,
            "CROP": crop,
            "CULTIVAR": cultivar,
            "YEAR": yr,
            "SYN": syn,
            "ORG": org,
        }
        row.update(others)
        unit_rows.append(row)

# 野菜
for (crop, cultivar), g in df_veg.groupby(["CROP", "CULTIVAR"]):
    g = g.sort_values("YEAR")
    a, b = linear_reg(g["YEAR"], g["SYN"])
    syn_2020 = a + b * 2020.0
    syn_2050 = 0.7 * syn_2020

    def syn_func(t):
        return syn_2020 + (syn_2050 - syn_2020) * ((t - 2020.0) / 30.0)

    n_req = g["N_REQ"].dropna().iloc[0] if g["N_REQ"].notna().any() else 0.0
    g_last = g.iloc[-1]
    others = {col: (0.0 if pd.isna(g_last[col]) else g_last[col])
              for col in ["AGROCHEM", "DIESEL", "GASOL", "KEROS", "ENGOIL", "ELECT"]}

    for yr in years_future:
        syn = max(0.0, float(syn_func(yr)))
        org = max(0.0, n_req - syn)
        row = {
            "CITY": np.nan,
            "CROP": crop,
            "CULTIVAR": cultivar,
            "YEAR": yr,
            "SYN": syn,
            "ORG": org,
        }
        row.update(others)
        unit_rows.append(row)

df_units = pd.DataFrame(unit_rows)
df_units["CITY"] = pd.to_numeric(df_units["CITY"], errors="coerce")

# 農業集落ごとの単位投入量配分
crop_rice_like = df_crop_rc[df_crop_rc["CROP_NORM"] != "tea"].copy()
crop_tea_like = df_crop_rc[df_crop_rc["CROP_NORM"] == "tea"].copy()

units_rice_like = df_units[df_units["CROP"] != "tea"].drop(columns=["CITY"])
merged_rice = crop_rice_like.merge(
    units_rice_like,
    on=["CROP", "CULTIVAR", "YEAR"],
    how="left"
)

units_tea_like = df_units[df_units["CROP"] == "tea"]
merged_tea = crop_tea_like.merge(
    units_tea_like,
    on=["CITY", "CROP", "CULTIVAR", "YEAR"],
    how="left"
)

merged = pd.concat([merged_rice, merged_tea], ignore_index=True)

for col in ["SYN", "ORG", "AGROCHEM", "DIESEL", "GASOL", "KEROS", "ENGOIL", "ELECT"]:
    merged[col] = merged[col].fillna(0.0)
for col in ["SYN", "ORG", "AGROCHEM", "DIESEL", "GASOL", "KEROS", "ENGOIL", "ELECT"]:
    merged[col + "_TOT"] = merged["EXTENT_PROJECTED"] * merged[col]
merged["CROP_NORM"] = merged["CROP_NORM"].astype(str).str.lower()

# 統合
def make_wide(df, index_cols, label):
    agg = df.groupby(index_cols + ["CROP_NORM"], dropna=False).agg(
        EXTENT=("EXTENT_PROJECTED", "sum"),
        SYN=("SYN_TOT", "sum"),
        ORG=("ORG_TOT", "sum"),
        AGROCHEM=("AGROCHEM_TOT", "sum"),
        DIESEL=("DIESEL_TOT", "sum"),
        GASOL=("GASOL_TOT", "sum"),
        KEROS=("KEROS_TOT", "sum"),
        ENGOIL=("ENGOIL_TOT", "sum"),
        ELECT=("ELECT_TOT", "sum")
    ).reset_index()

    wide = agg.pivot_table(
        index=index_cols,
        columns="CROP_NORM",
        values=["EXTENT", "SYN", "ORG", "AGROCHEM", "DIESEL", "GASOL", "KEROS", "ENGOIL", "ELECT"]
    )

    wide.columns = [
        f"{metric}_{crop.upper()}"
        for metric, crop in wide.columns
    ]
    wide = wide.reset_index()

    desired_cols = [
        *index_cols,
        "EXTENT_RICE", "EXTENT_TEA", "EXTENT_VEG",
        "SYN_RICE", "SYN_TEA", "SYN_VEG",
        "ORG_RICE", "ORG_TEA", "ORG_VEG",
        "AGROCHEM_RICE", "AGROCHEM_TEA", "AGROCHEM_VEG",
        "DIESEL_RICE", "DIESEL_TEA", "DIESEL_VEG",
        "GASOL_RICE", "GASOL_TEA", "GASOL_VEG",
        "KEROS_RICE", "KEROS_TEA", "KEROS_VEG",
        "ENGOIL_RICE", "ENGOIL_TEA", "ENGOIL_VEG",
        "ELECT_RICE", "ELECT_TEA", "ELECT_VEG",
    ]
    for col in desired_cols:
        if col not in wide.columns:
            wide[col] = np.nan

    wide = wide[desired_cols].copy()

    input_cols = [c for c in wide.columns if any(
        c.startswith(prefix) for prefix in ["SYN_", "ORG_", "AGROCHEM_", "DIESEL_", "GASOL_", "KEROS_", "ENGOIL_", "ELECT_"]
    )]
    wide[input_cols] = wide[input_cols].round(2)

    print(f"{label}: produced {len(wide)} rows")
    return wide

index_rc = ["KEY", "PREF", "CITY", "KCITY", "RCOM", "YEAR"]
wide_rcom = make_wide(merged, index_rc, label="RCOM-level")

agg_city = merged.groupby(
    ["PREF", "CITY", "YEAR", "CROP_NORM"], dropna=False
).agg(
    EXTENT_PROJECTED=("EXTENT_PROJECTED", "sum"),
    SYN_TOT=("SYN_TOT", "sum"),
    ORG_TOT=("ORG_TOT", "sum"),
    AGROCHEM_TOT=("AGROCHEM_TOT", "sum"),
    DIESEL_TOT=("DIESEL_TOT", "sum"),
    GASOL_TOT=("GASOL_TOT", "sum"),
    KEROS_TOT=("KEROS_TOT", "sum"),
    ENGOIL_TOT=("ENGOIL_TOT", "sum"),
    ELECT_TOT=("ELECT_TOT", "sum")
).reset_index()

city_meta = df_rcom_city.sort_values("KEY").groupby(["PREF", "CITY"]).first().reset_index()
agg_city = agg_city.merge(city_meta[["PREF", "CITY", "KEY"]], on=["PREF", "CITY"], how="left")
agg_city["KCITY"] = 0
agg_city["RCOM"] = 0

index_city = ["KEY", "PREF", "CITY", "KCITY", "RCOM", "YEAR"]
wide_city = make_wide(agg_city, index_city, label="CITY-level")

agg_pref = merged.groupby(
    ["PREF", "YEAR", "CROP_NORM"], dropna=False
).agg(
    EXTENT_PROJECTED=("EXTENT_PROJECTED", "sum"),
    SYN_TOT=("SYN_TOT", "sum"),
    ORG_TOT=("ORG_TOT", "sum"),
    AGROCHEM_TOT=("AGROCHEM_TOT", "sum"),
    DIESEL_TOT=("DIESEL_TOT", "sum"),
    GASOL_TOT=("GASOL_TOT", "sum"),
    KEROS_TOT=("KEROS_TOT", "sum"),
    ENGOIL_TOT=("ENGOIL_TOT", "sum"),
    ELECT_TOT=("ELECT_TOT", "sum")
).reset_index()

pref_meta = df_rcom_pref.sort_values("KEY").groupby("PREF").first().reset_index()
agg_pref = agg_pref.merge(pref_meta[["PREF", "KEY"]], on="PREF", how="left")
agg_pref["CITY"] = 0
agg_pref["KCITY"] = 0
agg_pref["RCOM"] = 0

index_pref = ["KEY", "PREF", "CITY", "KCITY", "RCOM", "YEAR"]
wide_pref = make_wide(agg_pref, index_pref, label="PREF-level")

agriculture_df = pd.concat([wide_rcom, wide_city, wide_pref], ignore_index=True)

agriculture_df.to_csv("01_agriculture.csv", index=False)
print("Saved 01_agriculture.csv (RCOM, CITY, PREF wide format).")

shapefile_path = "00_rcom.shp"

try:
    gdf = gpd.read_file(shapefile_path)

    for col in ["PREF", "CITY", "KCITY", "RCOM"]:
        if col in gdf.columns:
            gdf[col] = gdf[col].fillna(0).astype(int)
        if col in wide_rcom.columns:
            wide_rcom[col] = wide_rcom[col].fillna(0).astype(int)

    join_keys = ["PREF", "CITY", "KCITY", "RCOM"]

    gdf_merged = gdf.merge(
        wide_rcom,    # RCOM-level data only
        on=join_keys,
        how="left"
    )

    gdf_merged.to_file(
        "01_agriculture.gpkg",
        layer="agriculture",
        driver="GPKG"
    )

    print("GPKG export successful (RCOM level): 01_agriculture.gpkg")

except Exception as e:
    print("Agriculture GPKG export failed:", e)