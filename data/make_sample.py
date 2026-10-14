"""
Generate synthetic per-patient .psv files matching the real PhysioNet 2019
schema, so the pipeline runs offline. Replace with the real 42 MB download:

  https://physionet.org/content/challenge-2019/1.0.0/   (free, no credentialing)
  Point loaders at training/training_setA and training_setB.

What it copies from the real data:
  - heavy, channel-specific missingness (labs far sparser than vitals)
  - septic patients drifting abnormal in the hours before onset
  - the Challenge's label shift: SepsisLabel turns on 6h before onset
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from config import VITALS, LABS, DEMOGRAPHICS, LABEL, LABEL_SHIFT_HOURS  # noqa

RNG = np.random.default_rng(7)

# Rough normal baselines and per-hour measurement probabilities.
VITAL_BASE = {"HR": 80, "O2Sat": 97, "Temp": 36.9, "SBP": 120, "MAP": 82,
              "DBP": 64, "Resp": 16, "EtCO2": 33}
VITAL_SD = {"HR": 6, "O2Sat": 1.5, "Temp": 0.3, "SBP": 10, "MAP": 7,
            "DBP": 6, "Resp": 2, "EtCO2": 4}
VITAL_PMEAS = {"HR": .95, "O2Sat": .95, "Temp": .35, "SBP": .9, "MAP": .9,
               "DBP": .9, "Resp": .9, "EtCO2": .05}
LAB_BASE = {
    "BaseExcess": 0, "HCO3": 24, "FiO2": 0.5, "pH": 7.38, "PaCO2": 40,
    "SaO2": 96, "AST": 30, "BUN": 16, "Alkalinephos": 80, "Calcium": 9.2,
    "Chloride": 104, "Creatinine": 1.0, "Bilirubin_direct": 0.3,
    "Glucose": 130, "Lactate": 1.5, "Magnesium": 2.0, "Phosphate": 3.5,
    "Potassium": 4.1, "Bilirubin_total": 0.7, "TroponinI": 0.04, "Hct": 30,
    "Hgb": 10.3, "PTT": 34, "WBC": 9, "Fibrinogen": 280, "Platelets": 210,
}


def _one_patient(septic):
    length = int(RNG.integers(20, 61))
    age = float(np.clip(RNG.normal(63, 16), 18, 95))
    gender = int(RNG.integers(0, 2))
    unit1 = int(RNG.integers(0, 2))
    unit2 = 1 - unit1
    hosp_adm = float(-RNG.uniform(0, 60))

    # Onset (in hours) for septic patients; label starts 6h earlier.
    onset = int(RNG.integers(12, length)) if septic and length > 13 else None

    rows = []
    for t in range(length):
        row = {}
        sev = 0.0
        if onset is not None:
            sev = float(np.clip((t - (onset - 10)) / 10.0, 0, 1.2))

        for v in VITALS:
            mu = VITAL_BASE[v]
            if v in ("HR", "Resp", "Temp"):
                mu += sev * {"HR": 28, "Resp": 8, "Temp": 1.4}[v]
            if v in ("SBP", "MAP", "DBP", "O2Sat"):
                mu -= sev * {"SBP": 22, "MAP": 15, "DBP": 10, "O2Sat": 4}[v]
            val = RNG.normal(mu, VITAL_SD[v])
            row[v] = round(val, 1) if RNG.random() < VITAL_PMEAS[v] else np.nan

        for lab in LABS:
            base = LAB_BASE[lab]
            if lab in ("Lactate", "WBC", "Creatinine", "BUN"):
                base += sev * {"Lactate": 3.0, "WBC": 9, "Creatinine": 1.0,
                               "BUN": 12}[lab]
            p = 0.12 if lab in ("Lactate", "WBC", "Creatinine", "Glucose",
                                "Potassium", "BUN") else 0.05
            row[lab] = round(float(RNG.normal(base, abs(base) * 0.08 + 0.1)), 2) \
                if RNG.random() < p else np.nan

        row["Age"] = round(age, 1)
        row["Gender"] = gender
        row["Unit1"] = unit1
        row["Unit2"] = unit2
        row["HospAdmTime"] = round(hosp_adm, 2)
        row["ICULOS"] = t + 1
        if onset is not None and t >= onset - LABEL_SHIFT_HOURS:
            row[LABEL] = 1
        else:
            row[LABEL] = 0
        rows.append(row)

    return pd.DataFrame(rows)[VITALS + LABS + DEMOGRAPHICS + [LABEL]]


def make_sample(n_patients=400, septic_frac=0.09, out_dir="data/sample_psv"):
    os.makedirs(out_dir, exist_ok=True)
    n_septic = 0
    for i in range(n_patients):
        septic = RNG.random() < septic_frac
        n_septic += int(septic)
        df = _one_patient(septic)
        df.to_csv(os.path.join(out_dir, f"p{i:05d}.psv"), sep="|", index=False)
    print(f"Wrote {n_patients} patient files to {out_dir}/ "
          f"({n_septic} septic, {n_septic / n_patients:.1%})")


if __name__ == "__main__":
    make_sample()
