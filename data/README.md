# Data Inputs

This directory documents **external data inputs** used by the pipeline.  
Some files are **not included in this repository** due to size, licensing, or redistribution constraints.

---

## Census Block Group Crosswalk (2010 → 2020)

### Why a crosswalk is required

- **ACS 2019 5-year** estimates align to **2010 Census block group geography**
- **ACS 2020+** and **LODES8** align to **2020 Census block group geography**

To support **valid multi-year comparisons**, older ACS estimates must be re-expressed on 2020 block groups.  
This pipeline does so using an **areal-weighted crosswalk** from 2010 blocks to 2020 block groups.

---

## Source

The crosswalk used by this pipeline can be obtained from:

**IPUMS NHGIS (National Historical Geographic Information System)**  
https://www.nhgis.org/

After creating a free IPUMS account, download a crosswalk that maps:
- **2010 Census blocks** → **2020 Census block groups**

Typical NHGIS crosswalk files include overlap or proportion fields that allow areal weighting.

> ⚠️ Licensing note  
> NHGIS/IPUMS data products may have redistribution restrictions.  
> For that reason, **crosswalk files are intentionally excluded from this GitHub repository**.

---

## Expected file location

Place the crosswalk file at:

```text
data/crosswalk/tex_blk2010_bg2020.csv
