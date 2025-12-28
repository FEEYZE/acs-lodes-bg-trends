# ACS + LODES Block Group Trends Pipeline (2010→2020 Crosswalk + AGOL Publish)

A reproducible Python pipeline that pulls **ACS 5-Year** block group demographics and **LODES** workplace area characteristics, harmonizes vintages using a **2010→2020 Census Block Group crosswalk**, calculates corridor-ready metrics (e.g., **population/job density**, age and poverty indicators), and publishes the result to **ArcGIS Online** for use in **Web Maps, Dashboards, and StoryMaps**.

---

## Why this exists (the problem)

Marketing and planning teams often rely on “legacy assumptions” about where demand and opportunity are growing. This workflow was built to replace assumptions with **current, data-backed corridor insights** by comparing two time periods and visualizing **where population and job patterns are increasing or declining**—including breakdowns by age and related equity indicators.

**Result:** A single hosted GIS layer that can be symbolized and filtered by year/snapshot and used directly in web mapping products.

---

## What it produces

### Primary outputs
- A **feature layer** at the **Census Block Group (2020)** level with:
  - ACS demographics (population, race proxy, poverty, seniors, age buckets, etc.)
  - LODES jobs (total + wage bins)
  - Density metrics (jobs/acre, pop/acre, combined density index)
  - Two “row types” per block group:
    - **BASE**: years where ACS and LODES overlap (e.g., 2019)
    - **LATEST SNAPSHOT**: latest ACS year joined with latest LODES year (e.g., 2023 + 2022)

### Local files
- `outputs/jobs_and_population_combined_multi_year.shp`
- `outputs/jobs_and_population_combined_multi_year.csv`

---

## Data sources

- **ACS 5-Year (block group)** via Census API  
  https://api.census.gov/data.html
- **LODES8 WAC (Workplace Area Characteristics)**  
  https://lehd.ces.census.gov/data/
- **TIGER/Line Block Groups (2020 vintage BGs)**  
  https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html
- **NHGIS / crosswalk file** for **2010 BG → 2020 BG** harmonization  
  (Crosswalk file is referenced locally; see “Crosswalk” section below.)

---

## Key Features (More than “just downloading data”)

### ✅ Multi-dataset integration
Combines ACS demographics and LODES job data into a single GIS-ready table keyed to 2020 Census block groups.

### ✅ Vintage harmonization (2010→2020 BG crosswalk)
ACS 2019 5-year estimates align to **2010** block group definitions. This pipeline:
- joins ACS 2019 BGs to a 2010→2020 crosswalk
- **reweights** person-based counts using normalized overlap weights
- outputs standardized attributes on **2020 block groups**, enabling apples-to-apples trend comparisons

### ✅ GIS-ready metrics for corridor analysis
Creates area-based densities (using a projected CRS):
- `Jobs_per_acre`
- `Pop_per_acre`
- `pop_job_den` (combined index)

### ✅ AGOL automation (publish/overwrite)
Packages the output shapefile into a zip and:
- overwrites an existing **Hosted Feature Layer**, or
- publishes a new Hosted Feature Layer (configurable)

---
