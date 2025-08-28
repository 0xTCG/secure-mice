# -*- coding: utf-8 -*-
# MIMIC-IV: build analysis table with ICU type, age, first-2h vitals/labs,
#           normalized height (cm), weight (kg), and outcome death_within_48h.

import pandas as pd
from pathlib import Path

# ========= CONFIG =========
PATH = Path("MIMIC")  # <-- change if needed
OUT_CSV = PATH / "icu_first2h_features.csv"

# ========= HELPERS =========
def need(name, **kw):
    p = PATH / name
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")
    return pd.read_csv(p, **kw)

def opt(name, **kw):
    p = PATH / name
    return pd.read_csv(p, **kw) if p.exists() else None

def short_unit(name):
    if pd.isna(name): return name
    s = str(name).lower()
    if "medical" in s and "icu" in s: return "MICU"
    if "surgical" in s and "icu" in s and "trauma" not in s: return "SICU"
    if "coronary" in s or "cardiac care" in s or "ccu" in s: return "CCU"
    if "cardiac surgery" in s or "csru" in s: return "CSRU"
    if "trauma" in s and "icu" in s: return "TSICU"
    if "neuro" in s and "icu" in s: return "Neuro ICU"
    return name

def pick_first_by_priority(df, stay_col, time_col, varname, priorities, value_col="valuenum"):
    sub = df[df["itemid"].isin(priorities)].copy()
    if sub.empty:
        return pd.DataFrame(columns=[stay_col, varname])
    rank = {iid: i for i, iid in enumerate(priorities)}
    sub["priority"] = sub["itemid"].map(rank)
    sub = sub.sort_values([stay_col, "priority", time_col])
    first = sub.groupby(stay_col, as_index=False).first()[[stay_col, value_col]]
    return first.rename(columns={value_col: varname})

def find_itemids(d_items, regex, linksto="chartevents", max_n=10):
    abbr = d_items.get("abbreviation")
    mask = (d_items["linksto"]==linksto) & (
        d_items["label"].str.contains(regex, case=False, na=False) |
        (abbr.astype(str).str.contains(regex, case=False, na=False) if abbr is not None else False)
    )
    return d_items.loc[mask, ["itemid","label"]].sort_values("itemid").head(max_n)["itemid"].tolist()

# ========= LOAD DATA =========
icustays    = need("icustays.csv",   parse_dates=["intime","outtime"])
admissions  = need("admissions.csv", parse_dates=["admittime","dischtime","deathtime"])
chartevents = need("chartevents.csv",parse_dates=["charttime"])
d_items     = need("d_items.csv")

# Optional files
patients     = opt("patients.csv")                                  # for age and dod
demo_filter  = opt("demo_subject_id.csv")                           # to subset
labevents    = opt("labevents.csv", parse_dates=["charttime"])      # for glucose, pH
d_labitems   = opt("d_labitems.csv")

# Ensure dodgeath fields are parsed if they exist
if patients is not None and "dod" in patients.columns:
    patients["dod"] = pd.to_datetime(patients["dod"], errors="coerce")
if "dod" in admissions.columns:
    admissions["dod"] = pd.to_datetime(admissions["dod"], errors="coerce")

# Optional subject filter
if demo_filter is not None and "subject_id" in demo_filter.columns:
    keep = set(demo_filter["subject_id"])
    icustays = icustays[icustays["subject_id"].isin(keep)]
    admissions = admissions[admissions["subject_id"].isin(keep)]
    if "subject_id" in chartevents.columns:
        chartevents = chartevents[chartevents["subject_id"].isin(keep)]
    if labevents is not None and "subject_id" in labevents.columns:
        labevents = labevents[labevents["subject_id"].isin(keep)]

# ========= ICU + ADMISSIONS + AGE =========
icustays = icustays[["subject_id","hadm_id","stay_id","first_careunit","last_careunit","intime","outtime","los"]]
core = icustays.merge(
    admissions[["subject_id","hadm_id","admittime","dischtime","deathtime"]],
    on=["subject_id","hadm_id"], how="left"
)

# Age (best available)
age_df = None
if patients is not None:
    if "anchor_age" in patients.columns:
        age_df = patients[["subject_id","anchor_age"]].rename(columns={"anchor_age":"age"})
    elif "dob" in patients.columns:
        tmp = core[["subject_id","hadm_id","admittime"]].merge(
            patients[["subject_id","dob"]], on="subject_id", how="left"
        )
        tmp["age"] = (pd.to_datetime(tmp["admittime"]) - pd.to_datetime(tmp["dob"])).dt.days / 365.2425
        age_df = tmp[["subject_id","hadm_id","age"]]
elif "admission_age" in admissions.columns:
    age_df = admissions[["subject_id","hadm_id","admission_age"]].rename(columns={"admission_age":"age"})

if age_df is not None:
    keys = [k for k in ["subject_id","hadm_id"] if k in age_df.columns]
    core = core.merge(age_df, on=keys, how="left")
else:
    core["age"] = pd.NA

# ========= VITAL ITEMIDs =========
CHOSEN_VITALS = {
    "heart_rate": [220045],
    "map":        [220052, 220181],      # invasive mean, then non-invasive mean
    "sbp":        [220050, 220179],
    "dbp":        [220051, 220180],
    "resp_rate":  [220210],
    "spo2":       [220277, 223769],
    "temperature":[223761, 223762],      # Celsius, Fahrenheit
    "fio2":       [223835],
    "gcs_eye":    [220739],
    "gcs_verbal": [223900],
    "gcs_motor":  [223901],
    "height":     [226707],              # height (often cm)
    "weight":     [226512, 226531],      # admit / last weight (kg)
    "cap_refill": [223951],              # capillary refill rate
}
TEMP_F_ITEMIDS = {223762}

# Regex fallbacks if needed
if not CHOSEN_VITALS["height"]:
    CHOSEN_VITALS["height"] = find_itemids(d_items, r"\bheight\b")
if not CHOSEN_VITALS["weight"]:
    CHOSEN_VITALS["weight"] = find_itemids(d_items, r"\bweight\b")
if not CHOSEN_VITALS["cap_refill"]:
    CHOSEN_VITALS["cap_refill"] = find_itemids(d_items, r"capillary\s*refill")

# ========= FIRST 2 HOURS OF CHARTEVENTS =========
ce = chartevents.merge(core[["stay_id","intime"]], on="stay_id", how="inner")
mask2h = (ce["charttime"] >= ce["intime"]) & (ce["charttime"] < ce["intime"] + pd.Timedelta(hours=2))
ce2h = ce.loc[mask2h & ce["valuenum"].notna(), ["stay_id","itemid","charttime","valuenum","valueuom"]].copy()

# Temperature -> Celsius
is_temp_f = ce2h["itemid"].isin(TEMP_F_ITEMIDS)
ce2h.loc[is_temp_f, "valuenum"] = (ce2h.loc[is_temp_f, "valuenum"] - 32) * 5.0 / 9.0
ce2h.loc[is_temp_f, "valueuom"] = "C"

# Height -> cm | Weight -> kg
def normalize_ht_wt(df):
    df = df.copy()
    if "valueuom" not in df.columns:
        return df
    # height
    ht_mask = df["itemid"].isin(CHOSEN_VITALS["height"])
    ht_u = df.loc[ht_mask, "valueuom"].astype(str).str.lower()
    in_mask = ht_mask & ht_u.isin(["in", "inch", "inches"])
    df.loc[in_mask, "valuenum"] = df.loc[in_mask, "valuenum"] * 2.54
    df.loc[in_mask, "valueuom"] = "cm"
    m_mask = ht_mask & ht_u.isin(["m", "meter", "meters"])
    df.loc[m_mask, "valuenum"] = df.loc[m_mask, "valuenum"] * 100.0
    df.loc[m_mask, "valueuom"] = "cm"
    # weight
    wt_mask = df["itemid"].isin(CHOSEN_VITALS["weight"])
    wt_u = df.loc[wt_mask, "valueuom"].astype(str).str.lower()
    lb_mask = wt_mask & wt_u.isin(["lb", "lbs", "pound", "pounds"])
    df.loc[lb_mask, "valuenum"] = df.loc[lb_mask, "valuenum"] * 0.45359237
    df.loc[lb_mask, "valueuom"] = "kg"
    g_mask = wt_mask & wt_u.isin(["g", "gram", "grams"])
    df.loc[g_mask, "valuenum"] = df.loc[g_mask, "valuenum"] / 1000.0
    df.loc[g_mask, "valueuom"] = "kg"
    return df

ce2h = normalize_ht_wt(ce2h)

# Wide table for charted variables
wide_chart = None
for var, prios in CHOSEN_VITALS.items():
    part = pick_first_by_priority(ce2h, "stay_id", "charttime", var, prios)
    wide_chart = part if wide_chart is None else wide_chart.merge(part, on="stay_id", how="outer")

# GCS total
for c in ["gcs_eye","gcs_verbal","gcs_motor"]:
    if c not in (wide_chart.columns if wide_chart is not None else []):
        if wide_chart is None:
            wide_chart = pd.DataFrame(columns=["stay_id", c])
        else:
            wide_chart[c] = pd.NA
wide_chart["gcs_total"] = wide_chart[["gcs_eye","gcs_verbal","gcs_motor"]].sum(axis=1, min_count=1)

# ========= LABS (0–2h): glucose, pH (if available) =========
def labs_first_2h(core_frame):
    if labevents is None or d_labitems is None:
        return pd.DataFrame(columns=["stay_id","glucose","ph"])
    labs = labevents.merge(core_frame[["subject_id","hadm_id","stay_id","intime"]],
                           on=["subject_id","hadm_id"], how="inner")
    mask = (labs["charttime"] >= labs["intime"]) & (labs["charttime"] < labs["intime"] + pd.Timedelta(hours=2))
    labs = labs.loc[mask & labs["valuenum"].notna(), ["stay_id","itemid","charttime","valuenum"]].copy()
    labs = labs.merge(d_labitems[["itemid","label"]], on="itemid", how="left")

    def lab_ids(regex):
        return d_labitems.loc[d_labitems["label"].str.contains(regex, case=False, na=False),
                              ["itemid","label"]].sort_values("itemid")

    glu_ids = lab_ids(r"\bglucose\b")["itemid"].tolist()
    ph_ids  = lab_ids(r"\bpH\b")["itemid"].tolist()

    def pick_first_lab(df, varname, itemids, prefer_contains=None):
        if df is None or df.empty or not itemids:
            return pd.DataFrame(columns=["stay_id", varname])
        sub = df[df["itemid"].isin(itemids)].copy()
        if sub.empty:
            return pd.DataFrame(columns(["stay_id", varname]))
        if prefer_contains is not None and "label" in sub.columns:
            sub["pref"] = ~sub["label"].str.contains(prefer_contains, case=False, na=False)
        else:
            sub["pref"] = False
        sub = sub.sort_values(["stay_id","pref","charttime"])
        first = sub.groupby("stay_id", as_index=False).first()[["stay_id","valuenum"]]
        return first.rename(columns={"valuenum": varname})

    glu = pick_first_lab(labs, "glucose", glu_ids)
    ph  = pick_first_lab(labs, "ph", ph_ids, prefer_contains="arterial")
    return glu.merge(ph, on="stay_id", how="outer")

wide_labs = labs_first_2h(core)

# ========= ASSEMBLE BASE TABLE =========
core["icu_type_short"] = core["first_careunit"].apply(short_unit)
final = core.merge(wide_chart, on="stay_id", how="left").merge(wide_labs, on="stay_id", how="left")
final = final.sort_values(["subject_id","stay_id"])

# Column order
cols = [
    "subject_id","hadm_id","stay_id","first_careunit","icu_type_short","intime","outtime","los","age",
    "heart_rate","sbp","dbp","map","resp_rate","temperature","spo2","fio2",
    "gcs_total","gcs_eye","gcs_verbal","gcs_motor",
    "cap_refill","height","weight","glucose","ph"
]
final = final[[c for c in cols if c in final.columns]]

# ========= OUTCOME: death within 48h of ICU intime =========
death_time = core["deathtime"].copy()  # from admissions

# Add admissions.dod if present (align by subject_id, hadm_id)
if "dod" in admissions.columns:
    core = core.merge(admissions[["subject_id","hadm_id","dod"]],
                      on=["subject_id","hadm_id"], how="left", suffixes=("", "_adm"))
    death_time = death_time.fillna(core["dod"])

# Add patients.dod if present (align by subject_id)
if patients is not None and "dod" in patients.columns:
    core = core.merge(patients[["subject_id","dod"]].rename(columns={"dod":"dod_pat"}),
                      on="subject_id", how="left")
    death_time = death_time.fillna(core["dod_pat"])

# Create binary indicator and merge
final = final.merge(
    pd.DataFrame({
        "stay_id": core["stay_id"],
        "death_within_48h": (
            death_time.notna() & (death_time <= core["intime"] + pd.Timedelta(hours=48))
        ).astype(int)
    }),
    on="stay_id", how="left"
)

# ========= SAVE =========
final.to_csv(OUT_CSV, index=False)
print(f"Saved to: {OUT_CSV}")
print(final.head(10))