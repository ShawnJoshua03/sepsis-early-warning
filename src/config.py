"""Schema and constants for the PhysioNet/CinC 2019 Sepsis Challenge.

The real data ships as one pipe-delimited .psv per patient, one row per hour.
Columns below are the exact 40 challenge columns in order.
"""

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]

LABS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]

# Age, Gender (0/1), Unit1/Unit2 (ICU type one-hot), HospAdmTime (hrs, negative),
# ICULOS (ICU length of stay in hours = the time index).
DEMOGRAPHICS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]

LABEL = "SepsisLabel"

CHANNELS = VITALS + LABS          # time-varying measurements
ALL_COLUMNS = VITALS + LABS + DEMOGRAPHICS + [LABEL]

# The challenge labels are already shifted: SepsisLabel = 1 starting 6 hours
# BEFORE clinical onset, so predicting the label == predicting sepsis 6h early.
LABEL_SHIFT_HOURS = 6

# Rolling window (hours) for trend features on vitals.
ROLL_WINDOW = 6
