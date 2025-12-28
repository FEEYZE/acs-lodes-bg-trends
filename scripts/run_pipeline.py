# scripts/run_pipeline.py
"""
ACS + LODES Block Group Trends Pipeline (Portfolio Version)

Primary output: GeoPackage (.gpkg) + CSV
Optional: Publish/overwrite to ArcGIS Online (AGOL)

Notes:
- Crosswalk files (IPUMS/NHGIS) are intentionally excluded from git. See data/README.md.
- For AGOL publishing, this script can generate a TEMP shapefile and zip it because
  AGOL publishing workflows are most consistent with zipped shapefiles.
"""

from __future__ import annotations

import os
import io
import time
import glob
import gzip
import zipfile
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import geopandas as gpd
import requests

# Optional AGOL deps only used if publishing enabled
try:
    from arcgis.gis import GIS
    from arcgis.features import FeatureLayerCollection
except Exception:
    GIS = None
    FeatureLayerCollection = None


# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("acs_lodes_pipeline")


# -----------------------------
# CONFIG (edit here for portfolio demo)
# -----------------------------
STATE_FIPS = "48"
COUNTIES_FIPS = ["48029"]  # 5-digit county GEOID(s) (state+county)

ACS_YEARS = [2023, 2019]     # ACS 5-year end years
LODES_YEARS = [2022, 2019]   # LODES years

# Crosswalk: 2010 blocks -> 2020 BGs (expected cols: blk2010ge, bg2020ge, parea)
CROSSWALK_CSV = Path("data/crosswalk/tex_blk2010_bg2020.csv")

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

TIGER_YEAR = 2024

# Output format
GPKG_NAME = "jobs_and_population_combined_multi_year.gpkg"
GPKG_LAYER = "bg_trends"
CSV_NAME = "jobs_and_population_combined_multi_year.csv"

# Publishing to AGOL (optional)
PUBLISH_ENABLED = False
TARGET_FEATURE_LAYER_ITEM_ID = ""  # set to overwrite; leave blank to publish new
PUBLISH_TITLE = "BG Trends (ACS + LODES)"
PUBLISH_TAGS = "ACS, LODES, Census Block Groups, GIS"
PUBLISH_SUMMARY = "Pipeline output for block-group trends and job density metrics."
AGOL_FOLDER = None  # e.g., "Portfolio"


# -----------------------------
# Secrets from env vars
# -----------------------------
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY", "")

AGOL_PORTAL_URL = os.getenv("AGOL_PORTAL_URL", "https://www.arcgis.com")
AGOL_USERNAME = os.getenv("AGOL_USERNAME", "")
AGOL_PASSWORD = os.getenv("AGOL_PASSWORD", "")


# -----------------------------
# Helpers
# -----------------------------
def _chunks(lst: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def download_lodes(year: int, counties_fips: List[str]) -> pd.DataFrame:
    """
    LODES8 TX WAC, filtered to selected counties by w_geocode prefix (state+county).
    Note: LODES8 uses 2020 blocks for all years (BG = w_geocode[:12] == 2020 BG).
    """
    url = f"https://lehd.ces.census.gov/data/lodes/LODES8/tx/wac/tx_wac_S000_JT00_{year}.csv.gz"
    log.info(f"Downloading LODES WAC {year} ...")
    r = requests.get(url, timeout=120)
    if r.status_code != 200:
        log.warning(f"LODES {year}: HTTP {r.status_code}")
        return pd.DataFrame()

    with gzip.open(io.BytesIO(r.content), mode="rt") as f:
        df = pd.read_csv(f, dtype={"w_geocode": str})

    df = df[df["w_geocode"].str.startswith(tuple(counties_fips))].copy()
    df = df[["w_geocode", "C000", "CE01", "CE02", "CE03"]].copy()
    df.columns = ["GEOID20", "Jobs", "JobsLess1250", "Jobs1251_3333", "JobsOver3333"]
    df["BGGEOID"] = df["GEOID20"].str[:12]
    df["lodes_year"] = year
    df["lodes_date"] = pd.to_datetime(f"{year}-01-01")

    out = (
        df.groupby(["BGGEOID", "lodes_year", "lodes_date"], as_index=False)
        .sum(numeric_only=True)
    )
    return out


def fetch_acs(year: int, state_fips: str, counties_fips: List[str], api_key: str) -> pd.DataFrame:
    """
    Fetch ACS 5-year block-group data for `year` (end year of the 5-year).
    Uses chunked variable requests to stay under the API var limit.
    """
    if not api_key:
        raise RuntimeError("Missing CENSUS_API_KEY. Set it as an environment variable.")

    var_codes = [
        "NAME", "B01003_001E", "B02001_002E", "B08301_001E", "B08301_010E", "B19013_001E",
        # B01001 2..49
        "B01001_002E","B01001_003E","B01001_004E","B01001_005E","B01001_006E","B01001_007E",
        "B01001_008E","B01001_009E","B01001_010E","B01001_011E","B01001_012E","B01001_013E",
        "B01001_014E","B01001_015E","B01001_016E","B01001_017E","B01001_018E","B01001_019E",
        "B01001_020E","B01001_021E","B01001_022E","B01001_023E","B01001_024E","B01001_025E",
        "B01001_026E","B01001_027E","B01001_028E","B01001_029E","B01001_030E","B01001_031E",
        "B01001_032E","B01001_033E","B01001_034E","B01001_035E","B01001_036E","B01001_037E",
        "B01001_038E","B01001_039E","B01001_040E","B01001_041E","B01001_042E","B01001_043E",
        "B01001_044E","B01001_045E","B01001_046E","B01001_047E","B01001_048E","B01001_049E",
        # poverty detail
        "C17002_001E","C17002_002E","C17002_003E"
    ]

    geo_clause = f"&for=block group:*&in=state:{state_fips}&in=county:*&in=tract:*"
    base_url = f"https://api.census.gov/data/{year}/acs/acs5"
    chunk_size = 45

    merged = None
    for vars_chunk in _chunks(var_codes, chunk_size):
        url = f"{base_url}?get={','.join(vars_chunk)}{geo_clause}&key={api_key}"
        r = requests.get(url, timeout=120)
        if r.status_code != 200:
            log.warning(f"ACS {year}: HTTP {r.status_code} for chunk starting {vars_chunk[:3]}")
            return pd.DataFrame()

        data = r.json()
        df_chunk = pd.DataFrame(data[1:], columns=data[0])

        if merged is None:
            merged = df_chunk
        else:
            merged = merged.merge(df_chunk, on=["state", "county", "tract", "block group"], how="left")

        time.sleep(0.2)

    if merged is None or merged.empty:
        return pd.DataFrame()

    merged["acs_year"] = year
    merged["acs_date"] = pd.to_datetime(f"{year}-01-01")
    merged["COGEOID"] = merged["state"] + merged["county"]
    merged = merged[merged["COGEOID"].isin(counties_fips)].copy()
    merged["BGGEOID_raw"] = merged["state"] + merged["county"] + merged["tract"] + merged["block group"]

    ren = {
        "B01003_001E":"Tot_Pop", "B02001_002E":"TotWhtPop", "B08301_001E":"Work_Pop",
        "B08301_010E":"Pub_Transit", "B19013_001E":"MHI",
        "B01001_002E":"Male_Tot", "B01001_003E":"Male<5", "B01001_004E":"Male_5_9",
        "B01001_005E":"Male_10_14","B01001_006E":"Male_15_17","B01001_007E":"Male_18_19",
        "B01001_008E":"Male_20","B01001_009E":"Male_21","B01001_010E":"Male_22_24",
        "B01001_011E":"Male_25_29","B01001_012E":"Male_30_34","B01001_013E":"Male_35_39",
        "B01001_014E":"Male_40_44","B01001_015E":"Male_45_49","B01001_016E":"Male_50_54",
        "B01001_017E":"Male_55_59","B01001_018E":"Male_60_61","B01001_019E":"Male_62_64",
        "B01001_020E":"Male_65_66","B01001_021E":"Male_67_69","B01001_022E":"Male_70_74",
        "B01001_023E":"Male_75_79","B01001_024E":"Male_80_84","B01001_025E":"Male85+",
        "B01001_026E":"Fem_Tot","B01001_027E":"Fem<5","B01001_028E":"Fem_5_9",
        "B01001_029E":"Fem_10_14","B01001_030E":"Fem_15_17","B01001_031E":"Fem_18_19",
        "B01001_032E":"Fem_20","B01001_033E":"Fem_21","B01001_034E":"Fem_22_24",
        "B01001_035E":"Fem_25_29","B01001_036E":"Fem_30_34","B01001_037E":"Fem_35_39",
        "B01001_038E":"Fem_40_44","B01001_039E":"Fem_45_49","B01001_040E":"Fem_50_54",
        "B01001_041E":"Fem_55_59","B01001_042E":"Fem_60_61","B01001_043E":"Fem_62_64",
        "B01001_044E":"Fem_65_66","B01001_045E":"Fem_67_69","B01001_046E":"Fem_70_74",
        "B01001_047E":"Fem_75_79","B01001_048E":"Fem_80_84","B01001_049E":"Fem85+",
        "C17002_001E":"PovStatDet","C17002_002E":"Less50Pov","C17002_003E":"50to99Pov"
    }
    df = merged.rename(columns=ren)

    # numeric conversions
    for c in ren.values():
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["TotMinPop"] = (df["Tot_Pop"] - df["TotWhtPop"]).clip(lower=0)
    df["PovLess100"] = df["Less50Pov"] + df["50to99Pov"]
    df["tot_senior_pop"] = df[[
        "Male_65_66","Male_67_69","Male_70_74","Male_75_79","Male_80_84","Male85+",
        "Fem_65_66","Fem_67_69","Fem_70_74","Fem_75_79","Fem_80_84","Fem85+"
    ]].sum(axis=1)
    df["MHI"] = pd.to_numeric(df["MHI"], errors="coerce").fillna(0).clip(lower=0)

    # percentage columns (optional but useful for dashboards)
    male_age = ["Male<5","Male_5_9","Male_10_14","Male_15_17","Male_18_19","Male_20","Male_21","Male_22_24",
                "Male_25_29","Male_30_34","Male_35_39","Male_40_44","Male_45_49","Male_50_54","Male_55_59",
                "Male_60_61","Male_62_64","Male_65_66","Male_67_69","Male_70_74","Male_75_79","Male_80_84","Male85+"]
    fem_age  = ["Fem<5","Fem_5_9","Fem_10_14","Fem_15_17","Fem_18_19","Fem_20","Fem_21","Fem_22_24",
                "Fem_25_29","Fem_30_34","Fem_35_39","Fem_40_44","Fem_45_49","Fem_50_54","Fem_55_59",
                "Fem_60_61","Fem_62_64","Fem_65_66","Fem_67_69","Fem_70_74","Fem_75_79","Fem_80_84","Fem85+"]

    for c in ["Male_Tot", "Fem_Tot"] + male_age + fem_age:
        if c in df.columns:
            df[f"{c}_pctTotPop"] = np.where(df["Tot_Pop"] > 0, (df[c] / df["Tot_Pop"]) * 100, 0)
    for c in male_age:
        if c in df.columns:
            df[f"{c}_pctMale"] = np.where(df["Male_Tot"] > 0, (df[c] / df["Male_Tot"]) * 100, 0)
    for c in fem_age:
        if c in df.columns:
            df[f"{c}_pctFem"] = np.where(df["Fem_Tot"] > 0, (df[c] / df["Fem_Tot"]) * 100, 0)

    return df


def load_bg_crosswalk(fpath: Path) -> pd.DataFrame:
    """
    Expected columns (minimum): blk2010ge, bg2020ge, parea
    Produces BG2010 -> BG2020 weights normalized by BG2010.
    """
    cw = pd.read_csv(fpath, dtype={"blk2010ge": str, "bg2020ge": str}, low_memory=False)
    for c in ["blk2010ge", "bg2020ge", "parea"]:
        if c not in cw.columns:
            raise KeyError(f"Crosswalk missing required column: {c}")

    cw["parea"] = pd.to_numeric(cw["parea"], errors="coerce").fillna(0.0)
    cw["BG2010"] = cw["blk2010ge"].str.slice(0, 12)
    cw["BG2020"] = cw["bg2020ge"].str.slice(0, 12)

    agg = (
        cw.groupby(["BG2010", "BG2020"], as_index=False)["parea"]
        .sum()
        .rename(columns={"parea": "overlap_area"})
    )

    totals = agg.groupby("BG2010", as_index=False)["overlap_area"].sum().rename(columns={"overlap_area": "_tot"})
    xwalk = agg.merge(totals, on="BG2010", how="left")
    xwalk["wt_area"] = np.where(xwalk["_tot"] > 0, xwalk["overlap_area"] / xwalk["_tot"], 0.0)

    # Using areal weights as a proxy for person/household weighting
    xwalk["wt_pop"] = xwalk["wt_area"]
    xwalk["wt_hh"] = xwalk["wt_area"]

    return xwalk[["BG2010", "BG2020", "wt_pop", "wt_hh"]]


def reweight_2010BG_to_2020BG(acs_df_2019: pd.DataFrame, crosswalk: pd.DataFrame) -> pd.DataFrame:
    df = acs_df_2019.copy().rename(columns={"BGGEOID_raw": "BG2010"})
    df = df.merge(crosswalk, on="BG2010", how="inner")

    person_vars = [
        "Tot_Pop","TotWhtPop","Work_Pop","Pub_Transit","PovStatDet","Less50Pov","50to99Pov",
        "TotMinPop","PovLess100","tot_senior_pop",
        "Male_Tot","Fem_Tot",
        "Male<5","Male_5_9","Male_10_14","Male_15_17","Male_18_19","Male_20","Male_21","Male_22_24",
        "Male_25_29","Male_30_34","Male_35_39","Male_40_44","Male_45_49","Male_50_54","Male_55_59",
        "Male_60_61","Male_62_64","Male_65_66","Male_67_69","Male_70_74","Male_75_79","Male_80_84","Male85+",
        "Fem<5","Fem_5_9","Fem_10_14","Fem_15_17","Fem_18_19","Fem_20","Fem_21","Fem_22_24",
        "Fem_25_29","Fem_30_34","Fem_35_39","Fem_40_44","Fem_45_49","Fem_50_54","Fem_55_59",
        "Fem_60_61","Fem_62_64","Fem_65_66","Fem_67_69","Fem_70_74","Fem_75_79","Fem_80_84","Fem85+"
    ]

    for c in person_vars:
        if c in df.columns:
            df[c] = df[c].astype(float) * df["wt_pop"].astype(float)

    # MHI "approx" weighted by wt_hh (areal proxy)
    df["_MHI_num"] = df["MHI"].astype(float) * df["wt_hh"].astype(float)

    agg_cols = {c: "sum" for c in person_vars + ["_MHI_num"]}
    df_agg = df.groupby(["BG2020", "acs_year", "acs_date"], as_index=False).agg(agg_cols)
    df_den = (
        df.groupby(["BG2020", "acs_year", "acs_date"], as_index=False)["wt_hh"]
        .sum()
        .rename(columns={"wt_hh": "_MHI_den"})
    )

    df_agg = df_agg.merge(df_den, on=["BG2020", "acs_year", "acs_date"], how="left")
    df_agg["MHI"] = np.where(df_agg["_MHI_den"] > 0, (df_agg["_MHI_num"] / df_agg["_MHI_den"]).round(), 0)

    df_agg = df_agg.rename(columns={"BG2020": "BGGEOID"}).drop(columns=["_MHI_den"], errors="ignore")

    # derived % metrics for dashboarding
    df_agg["MinPopPer"] = np.where(df_agg["Tot_Pop"] > 0, (df_agg["TotMinPop"] / df_agg["Tot_Pop"]) * 100, 0)
    df_agg["AtBelowPov"] = np.where(df_agg["PovStatDet"] > 0, (df_agg["PovLess100"] / df_agg["PovStatDet"]) * 100, 0)
    df_agg["Transit%"] = np.where(df_agg["Work_Pop"] > 0, (df_agg["Pub_Transit"] / df_agg["Work_Pop"]) * 100, 0)
    df_agg["Sen_Pop%"] = np.where(df_agg["Tot_Pop"] > 0, (df_agg["tot_senior_pop"] / df_agg["Tot_Pop"]) * 100, 0)

    return df_agg


def get_tiger_bg(state_fips: str, counties_fips: List[str], tiger_year: int) -> gpd.GeoDataFrame:
    tiger_url = f"https://www2.census.gov/geo/tiger/TIGER{tiger_year}/BG/tl_{tiger_year}_{state_fips}_bg.zip"
    tiger_zip = Path(f"tl_{tiger_year}_{state_fips}_bg.zip")
    tiger_dir = Path(f"tl_{tiger_year}_{state_fips}_bg")

    if not tiger_zip.exists():
        log.info("Downloading TIGER BG shapefile...")
        tiger_zip.write_bytes(requests.get(tiger_url, timeout=120).content)

    if not tiger_dir.exists():
        with zipfile.ZipFile(tiger_zip, "r") as z:
            z.extractall(tiger_dir)

    shp_path = glob.glob(str(tiger_dir / "*.shp"))[0]
    gdf = gpd.read_file(shp_path)

    # county fips: TIGER has COUNTYFP (3-digit), our counties_fips are 5-digit (state+county)
    countyfp = [c[2:] for c in counties_fips]
    gdf = gdf[gdf["COUNTYFP"].isin(countyfp)].rename(columns={"GEOID": "BGGEOID"})
    return gdf


def zip_shapefile(shp_path: Path) -> Path:
    """
    Zips all shapefile sidecars that share the same basename.
    """
    base = shp_path.with_suffix("")
    zip_path = base.with_suffix(".zip")

    exts = [".shp", ".shx", ".dbf", ".prj", ".cpg", ".sbn", ".sbx", ".qix", ".fix", ".shp.xml"]
    files = [base.with_suffix(ext) for ext in exts if base.with_suffix(ext).exists()]
    if not files:
        raise FileNotFoundError(f"No shapefile components found for: {shp_path}")

    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, arcname=f.name)

    return zip_path


def agol_login() -> "GIS":
    if GIS is None:
        raise RuntimeError("arcgis package not available. Install arcgis or disable publishing.")
    if not (AGOL_USERNAME and AGOL_PASSWORD):
        raise RuntimeError("Missing AGOL_USERNAME / AGOL_PASSWORD env vars.")
    return GIS(AGOL_PORTAL_URL, AGOL_USERNAME, AGOL_PASSWORD)


def overwrite_hosted_feature_layer(gis: "GIS", fl_item_id: str, zipped_shp: Path) -> str:
    if FeatureLayerCollection is None:
        raise RuntimeError("arcgis.features.FeatureLayerCollection not available.")
    item = gis.content.get(fl_item_id)
    if item is None:
        raise ValueError(f"Could not find AGOL item: {fl_item_id}")

    flc = FeatureLayerCollection.fromitem(item)
    flc.manager.overwrite(str(zipped_shp))

    item.update(item_properties={"tags": PUBLISH_TAGS, "snippet": PUBLISH_SUMMARY})
    return item.id


def publish_new_hosted_feature_layer(gis: "GIS", zipped_shp: Path, title: str) -> str:
    shp_item = gis.content.add(
        item_properties={"type": "Shapefile", "title": title, "tags": PUBLISH_TAGS, "snippet": PUBLISH_SUMMARY},
        data=str(zipped_shp),
        folder=AGOL_FOLDER
    )
    published = shp_item.publish()
    return published.id


def main() -> None:
    # 1) LODES
    lodes_all = [download_lodes(y, COUNTIES_FIPS) for y in LODES_YEARS]
    lodes_df = pd.concat(lodes_all, ignore_index=True) if lodes_all else pd.DataFrame()

    # 2) ACS (reweight when pre-2020 vintage)
    acs_frames = []
    cw = None

    for y in ACS_YEARS:
        log.info(f"Fetching ACS {y} ...")
        dfy = fetch_acs(y, STATE_FIPS, COUNTIES_FIPS, CENSUS_API_KEY)
        if dfy.empty:
            continue

        if y <= 2019:
            if cw is None:
                if not CROSSWALK_CSV.exists():
                    raise FileNotFoundError(
                        f"Crosswalk not found: {CROSSWALK_CSV}\n"
                        "See data/README.md for where to obtain this file."
                    )
                cw = load_bg_crosswalk(CROSSWALK_CSV)
            dfy_std = reweight_2010BG_to_2020BG(dfy, cw)
            acs_frames.append(dfy_std)
        else:
            dfy = dfy.rename(columns={"BGGEOID_raw": "BGGEOID"})
            dfy["MinPopPer"] = np.where(dfy["Tot_Pop"] > 0, (dfy["TotMinPop"] / dfy["Tot_Pop"]) * 100, 0)
            dfy["AtBelowPov"] = np.where(dfy["PovStatDet"] > 0, (dfy["PovLess100"] / dfy["PovStatDet"]) * 100, 0)
            dfy["Transit%"] = np.where(dfy["Work_Pop"] > 0, (dfy["Pub_Transit"] / dfy["Work_Pop"]) * 100, 0)
            dfy["Sen_Pop%"] = np.where(dfy["Tot_Pop"] > 0, (dfy["tot_senior_pop"] / dfy["Tot_Pop"]) * 100, 0)
            acs_frames.append(dfy)

    acs_df = pd.concat(acs_frames, ignore_index=True) if acs_frames else pd.DataFrame()

    # 3) TIGER BG geometry
    gdf = get_tiger_bg(STATE_FIPS, COUNTIES_FIPS, TIGER_YEAR)

    # 4) Harmonize: BASE rows for shared years + latest snapshot (ACS latest + LODES latest)
    if not acs_df.empty:
        acs_df = acs_df.drop_duplicates(subset=["BGGEOID", "acs_year"])
    if not lodes_df.empty:
        lodes_df = lodes_df.drop_duplicates(subset=["BGGEOID", "lodes_year"])

    base_rows = []
    common_years = sorted(set(acs_df["acs_year"].unique()).intersection(lodes_df["lodes_year"].unique()))
    for yr in common_years:
        acs_y = acs_df[acs_df["acs_year"] == yr].copy()
        lodes_y = lodes_df[lodes_df["lodes_year"] == yr].copy()

        base_y = acs_y.merge(
            lodes_y[["BGGEOID", "lodes_year", "lodes_date", "Jobs", "JobsLess1250", "Jobs1251_3333", "JobsOver3333"]],
            on="BGGEOID",
            how="left",
        )
        base_y["lodes_year"] = yr
        base_y["lodes_date"] = pd.to_datetime(f"{yr}-01-01")
        for c in ["Jobs", "JobsLess1250", "Jobs1251_3333", "JobsOver3333"]:
            base_y[c] = pd.to_numeric(base_y.get(c, 0), errors="coerce").fillna(0).astype(int)

        base_rows.append(base_y)

    base_panel = pd.concat(base_rows, ignore_index=True) if base_rows else acs_df.iloc[0:0].copy()

    acs_latest_year = int(acs_df["acs_year"].max()) if not acs_df.empty else None
    lodes_latest_year = int(lodes_df["lodes_year"].max()) if not lodes_df.empty else None

    snapshot = acs_df[acs_df["acs_year"] == acs_latest_year].copy()
    lodes_target = lodes_df[lodes_df["lodes_year"] == lodes_latest_year][
        ["BGGEOID", "lodes_year", "lodes_date", "Jobs", "JobsLess1250", "Jobs1251_3333", "JobsOver3333"]
    ].copy()

    snapshot = snapshot.merge(lodes_target, on="BGGEOID", how="left")
    if lodes_latest_year is not None:
        snapshot["lodes_year"] = lodes_latest_year
        snapshot["lodes_date"] = pd.to_datetime(f"{lodes_latest_year}-01-01")
    for c in ["Jobs", "JobsLess1250", "Jobs1251_3333", "JobsOver3333"]:
        snapshot[c] = pd.to_numeric(snapshot.get(c, 0), errors="coerce").fillna(0).astype(int)

    combined = pd.concat([base_panel, snapshot], ignore_index=True)
    combined = combined.drop_duplicates(subset=["BGGEOID", "acs_year", "lodes_year"], errors="ignore")

    # 5) Merge geometry + compute densities
    final_gdf = gdf.merge(combined, on="BGGEOID", how="left")
    numeric_cols = final_gdf.select_dtypes(include=[np.number]).columns
    final_gdf[numeric_cols] = final_gdf[numeric_cols].fillna(0)

    # area/density calculations in a projected CRS
    final_gdf = final_gdf.to_crs(epsg=3081)  # Texas Centric Albers (good for area)
    final_gdf["Area_sqm"] = final_gdf.geometry.area
    final_gdf["Area_acres"] = (final_gdf["Area_sqm"] / 4046.8564224).replace(0, np.nan)

    for col, out in [
        ("Jobs", "Jobs_per_acre"),
        ("JobsLess1250", "JobsLess1250_per_acre"),
        ("Jobs1251_3333", "Jobs1251_3333_per_acre"),
        ("JobsOver3333", "JobsOver3333_per_acre"),
        ("Tot_Pop", "Pop_per_acre"),
    ]:
        if col in final_gdf.columns:
            final_gdf[out] = (final_gdf[col] / final_gdf["Area_acres"]).fillna(0)

    if "Jobs_per_acre" in final_gdf.columns and "Pop_per_acre" in final_gdf.columns:
        final_gdf["pop_job_den"] = final_gdf["Jobs_per_acre"] + final_gdf["Pop_per_acre"]

    # pretty date strings
    for date_col in ["lodes_date", "acs_date"]:
        if date_col in final_gdf.columns:
            final_gdf[date_col] = pd.to_datetime(final_gdf[date_col], errors="coerce").dt.strftime("%Y-%m-%d")

    # -----------------------------
    # EXPORT (GeoPackage + CSV)
    # -----------------------------
    gpkg_out = OUTPUTS_DIR / GPKG_NAME
    csv_out = OUTPUTS_DIR / CSV_NAME

    # overwrite existing layer in the gpkg by deleting file (simplest + cleanest for portfolio)
    if gpkg_out.exists():
        gpkg_out.unlink()

    log.info(f"Writing GeoPackage: {gpkg_out} (layer={GPKG_LAYER})")
    final_gdf.to_file(gpkg_out, driver="GPKG", layer=GPKG_LAYER)

    log.info(f"Writing CSV: {csv_out}")
    final_gdf.drop(columns="geometry").to_csv(csv_out, index=False)

    log.info("✅ Local outputs created (GeoPackage + CSV).")

    # -----------------------------
    # PUBLISH / OVERWRITE TO AGOL (OPTIONAL)
    # -----------------------------
    if not PUBLISH_ENABLED:
        log.info("Publishing disabled (PUBLISH_ENABLED=False). Done.")
        return

    log.info("Publishing enabled. Creating a TEMP shapefile for AGOL upload...")
    tmp_dir = Path(tempfile.mkdtemp(prefix="agol_publish_"))
    try:
        shp_base = tmp_dir / "bg_trends_publish"
        shp_path = shp_base.with_suffix(".shp")

        # AGOL upload is most consistent with zipped shapefile
        final_gdf.to_file(shp_path)

        shp_zip = zip_shapefile(shp_path)
        log.info(f"Prepared AGOL zip: {shp_zip}")

        log.info("Logging into AGOL...")
        gis = agol_login()
        log.info(f"Logged in as: {gis.users.me.username}")

        if TARGET_FEATURE_LAYER_ITEM_ID:
            log.info("Overwriting existing hosted feature layer...")
            item_id = overwrite_hosted_feature_layer(gis, TARGET_FEATURE_LAYER_ITEM_ID, shp_zip)
            log.info(f"✅ Overwrite complete. Item ID: {item_id}")
        else:
            log.info("Publishing new hosted feature layer...")
            item_id = publish_new_hosted_feature_layer(gis, shp_zip, PUBLISH_TITLE)
            log.info(f"✅ Publish complete. Item ID: {item_id}")

    finally:
        # clean up temp files (even if AGOL publish fails)
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            log.info("Cleaned up temporary publish files.")
        except Exception:
            pass


if __name__ == "__main__":
    main()
