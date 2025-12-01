import pandas as pd
import geopandas as gpd

rcom = gpd.read_file("00_rcom.shp")
sindex = rcom.sindex
neighbors_dict = {}

for idx, geom in rcom["geometry"].items():
    if geom is None:
        neighbors_dict[idx] = []
        continue

    possible_matches_index = list(sindex.intersection(geom.bounds))
    neighbors = [
        j for j in possible_matches_index
        if j != idx and rcom.at[j, "geometry"].touches(geom)
    ]
    neighbors_dict[idx] = neighbors

key_cols = ["KEY", "PREF", "CITY", "KCITY", "RCOM"]

# 気候データ
df_climate = pd.read_csv("00_climate.csv")
climate_vars = ["AIRTEMP", "PRECIP", "WIND"]
df_climate_clean = df_climate.dropna(subset=climate_vars).copy()
for v in climate_vars:
    df_climate_clean[v] = df_climate_clean[v].round(2)

# メッシュに変換
gdf_climate = gpd.GeoDataFrame(
    df_climate_clean,
    geometry=gpd.points_from_xy(df_climate_clean["LON"], df_climate_clean["LAT"]),
    crs="EPSG:4326"
)

# 農業集落ごと
if gdf_climate.crs != rcom.crs:
    gdf_climate = gdf_climate.to_crs(rcom.crs)

points_with_poly = gpd.sjoin(
    gdf_climate,
    rcom[["geometry"]],
    how="left",
    predicate="within"
)

agg_climate = (
    points_with_poly.dropna(subset=["index_right"])
    .groupby("index_right")[climate_vars]
    .mean()
)

rcom_climate = rcom.join(agg_climate, how="left")

# 空白地解決
for var in climate_vars:
    nan_idx = rcom_climate.index[rcom_climate[var].isna()]
    for idx in nan_idx:
        neigh = neighbors_dict.get(idx, [])
        if not neigh:
            continue
        neigh_vals = rcom_climate.loc[neigh, var].dropna()
        if neigh_vals.empty:
            continue
        val = float(neigh_vals.mean())
        rcom_climate.at[idx, var] = round(val, 2)

rcom_climate[climate_vars] = rcom_climate[climate_vars].round(2)

# 土壌データ
df_soil = pd.read_csv("00_soil.csv")

soil_vars = ["SOC", "SAND", "pH", "CEC", "NSTOCK", "SOILTEMP"]

df_soil_clean = df_soil.dropna(subset=soil_vars).copy()
for v in soil_vars:
    df_soil_clean[v] = df_soil_clean[v].round(2)

# メッシュに変換
gdf_soil = gpd.GeoDataFrame(
    df_soil_clean,
    geometry=gpd.points_from_xy(df_soil_clean["LON"], df_soil_clean["LAT"]),
    crs="EPSG:4326"
)

# 農業集落ごと
if gdf_soil.crs != rcom.crs:
    gdf_soil = gdf_soil.to_crs(rcom.crs)

soil_with_poly = gpd.sjoin(
    gdf_soil,
    rcom[["geometry"]],
    how="left",
    predicate="within"
)

soil_agg = (
    soil_with_poly.dropna(subset=["index_right"])
    .groupby("index_right")[soil_vars]
    .mean()
)

rcom_soil = rcom.join(soil_agg, how="left")

# 空白地解決
for var in soil_vars:
    nan_idx = rcom_soil.index[rcom_soil[var].isna()]
    for idx in nan_idx:
        neigh = neighbors_dict.get(idx, [])
        if not neigh:
            continue
        neigh_vals = rcom_soil.loc[neigh, var].dropna()
        if neigh_vals.empty:
            continue
        val = float(neigh_vals.mean())
        rcom_soil.at[idx, var] = round(val, 2)

rcom_soil[soil_vars] = rcom_soil[soil_vars].round(2)

# 統合
soil_columns = key_cols + soil_vars
soil_attr = rcom_soil[soil_columns].copy()

rcom_env = rcom_climate.merge(
    soil_attr,
    on=key_cols,
    how="left",
    suffixes=("", "_SOIL")
)

env_vars = climate_vars + soil_vars

rcom_env[env_vars] = rcom_env[env_vars].round(2)

rcom_env.to_file("01_env.gpkg", layer="env", driver="GPKG")
print("GPKG export successful (RCOM level): 01_env.gpkg")

env_columns = key_cols + env_vars
key_env = rcom_env[env_columns].copy()
key_env.to_csv("01_env.csv", index=False)
print("Saved 01_env.csv (RCOM climate + soil).")