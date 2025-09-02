import re
from typing import List
import numpy as np
import pandas as pd

def load_lnp_dataset(path_to_csv: str) -> pd.DataFrame:
    """
    Load & preprocess a lipid nanoparticle (LNP) formulation dataset from CSV.

    Parameters
    ----------
    path_to_csv : str
        File path to the dataset CSV.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset with:
          - snake_case column names
          - whitespace-trimmed strings
          - numeric columns coerced from strings-with-units
          - standardized key LNP fields (ee_frac in [0,1], size_nm, pdi, zeta_mv, conc_mg_ml, n_p_ratio)
          - duplicates removed and obvious outliers clipped to sane bounds (light-touch)

    """

    # ---------- helpers ----------
    def to_snake(name: str) -> str:
        name = name.strip()
        name = re.sub(r"[^\w\s]", " ", name)
        name = re.sub(r"\s+", "_", name)
        return name.lower()

    def strip_strings(df_: pd.DataFrame) -> pd.DataFrame:
        obj_cols = df_.select_dtypes(include=["object"]).columns
        for c in obj_cols:
            df_[c] = df_[c].astype(str).str.strip()
            # collapse internal whitespace
            df_[c] = df_[c].str.replace(r"\s+", " ", regex=True)
        return df_

    def parse_numeric(val):
        """
        Extract the first float-like number from a raw value that may contain units.
        Returns np.nan if nothing parseable is found.
        """
        if pd.isna(val):
            return np.nan
        s = str(val).strip()
        # Common 'ND', 'NA', '-', etc.
        if s.lower() in {"na", "nan", "none", "null", "-", ""}:
            return np.nan
        # Extract first signed float (supports sci notation)
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        return float(m.group(0)) if m else np.nan

    def parse_percent_to_frac(val):
        """
        Convert strings like '92%', '92 percent', '0.92' to a fraction in [0,1].
        Numeric rules:
          - If value > 1 and <= 100 → assume percent, divide by 100.
          - If value <= 1 → assume already fraction.
          - If value > 100 → treat as NaN (likely bad entry).
        """
        x = parse_numeric(val)
        if np.isnan(x):
            return np.nan
        if x > 1.0 and x <= 100.0:
            return x / 100.0
        if 0.0 <= x <= 1.0:
            return x
        return np.nan

    def coerce_numeric_col(df_: pd.DataFrame, col: str):
        if col in df_.columns:
            df_[col] = df_[col].map(parse_numeric)

    def coerce_percent_col(df_: pd.DataFrame, col: str, out_col: str):
        if col in df_.columns:
            df_[out_col] = df_[col].map(parse_percent_to_frac)

    # ---------- load ----------
    df = pd.read_csv(path_to_csv)

    # ---------- normalize headers & strings ----------
    df.columns = [to_snake(c) for c in df.columns]
    df = strip_strings(df)

    # ---------- alias resolution ---------- #### TO BE ADJUSTED WITH REAL DATA
    # Map common LNP names to standardized keys if present
    ALIASES = {
        # size
        "size_nm": ["size", "particle_size", "hydrodynamic_size", "diameter_nm", "z_avg_nm", "zave_nm"],
        # pdi
        "pdi": ["polydispersity", "polydispersity_index", "pdi_value"],
        # zeta potential (mV)
        "zeta_mv": ["zeta", "zeta_potential", "zeta_potential_mv"],
        # encapsulation efficiency
        "ee_percent": ["encapsulation_efficiency", "ee", "encap_eff", "encapsulation_%", "ee_%"],
        # concentration mg/mL
        "conc_mg_ml": ["concentration", "conc", "lipid_concentration", "total_lipid_mg_ml"],
        # N/P ratio
        "n_p_ratio": ["n_p", "n_to_p", "npratio", "n_p_molar_ratio"],
        # pH
        "buffer_ph": ["ph", "buffer_p_h"],
        # temperature C
        "temp_c": ["temperature", "temp", "temperature_c"],
        # flow rate (mL/min)
        "flow_ml_min": ["flow_rate", "flowrate", "flow_ml_per_min"],
    }

    # Promote first found alias to the canonical name
    for canonical, candidates in ALIASES.items():
        for c in candidates:
            if c in df.columns and canonical not in df.columns:
                df = df.rename(columns={c: canonical})
                break

    # ---------- numeric coercions ----------
    # Unit-bearing numerics
    for candidate in ["size_nm", "zeta_mv", "conc_mg_ml", "n_p_ratio", "buffer_ph", "temp_c", "flow_ml_min"]:
        coerce_numeric_col(df, candidate)

    # Percent-like → fraction [0,1]
    if "ee_percent" in df.columns:
        coerce_percent_col(df, "ee_percent", "ee_frac")

    # If someone stored EE already as 'ee' but without % in the name, try to detect
    if "ee" in df.columns and "ee_frac" not in df.columns:
        coerce_percent_col(df, "ee", "ee_frac")

    # ---------- light-touch sanity clipping ----------
    # PDI typically ~ 0–1.5
    if "pdi" in df.columns:
        df["pdi"] = df["pdi"].map(parse_numeric)
        df.loc[(df["pdi"] < 0) | (df["pdi"] > 2), "pdi"] = np.nan

    # Size nm: clip absurd values (keep 10–1000nm as plausible envelope)
    if "size_nm" in df.columns:
        df.loc[(df["size_nm"] < 10) | (df["size_nm"] > 10000), "size_nm"] = np.nan

    # Zeta (mV): clip |zeta| > 200mV as implausible instrument readouts
    if "zeta_mv" in df.columns:
        df.loc[df["zeta_mv"].abs() > 200, "zeta_mv"] = np.nan

    # EE fraction must be 0–1
    if "ee_frac" in df.columns:
        df.loc[(df["ee_frac"] < 0) | (df["ee_frac"] > 1), "ee_frac"] = np.nan

    # pH: clip to 0–14
    if "buffer_ph" in df.columns:
        df.loc[(df["buffer_ph"] < 0) | (df["buffer_ph"] > 14), "buffer_ph"] = np.nan

    # ---------- deduplicate & tidy ----------
    # If there’s an obvious formulation id, deduplicate on it; otherwise on all columns
    id_like: List[str] = [c for c in df.columns if c in {"formulation_id", "batch_id", "sample_id", "id"}]
    if id_like:
        df = df.drop_duplicates(subset=id_like, keep="first")
    else:
        df = df.drop_duplicates(keep="first")

    # Replace common missing markers with NaN (already mostly handled)
    df = df.replace({"": np.nan, "NA": np.nan, "NaN": np.nan, "None": np.nan})

    # Reorder: surface the most-model-relevant columns if present
    front = [c for c in [
        "formulation_id", "batch_id", "sample_id", "id",
        "size_nm", "pdi", "zeta_mv", "ee_frac", "conc_mg_ml",
        "n_p_ratio", "buffer_ph", "temp_c", "flow_ml_min"
    ] if c in df.columns]
    others = [c for c in df.columns if c not in front]
    df = df[front + others].reset_index(drop=True)

    return df
