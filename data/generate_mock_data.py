"""
MedPredict AI - Clinical Data Generator v6
--------------------------------------------
v6 key improvements:
1. Better class balance: Diabetes ~45%, Heart ~35%, Liver ~40%
2. Liver Disease: strong labeling from ALT/AST with clear hard overrides
3. All three diseases support comorbidities naturally
4. Smooth gradient of low → medium → high risk patients
5. 15,000 total samples for robust learning
"""

import pandas as pd
import numpy as np
import os

SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
#  CORE LABEL ASSIGNMENT  (runs on RAW unscaled feature values)
# ─────────────────────────────────────────────────────────────────────────────
def assign_labels(df: pd.DataFrame, rng) -> pd.DataFrame:
    n = len(df)

    age   = df['Age'].values.astype(float)
    bmi   = df['BMI'].values.astype(float)
    smk   = df['Smoking'].values.astype(float)
    pa    = df['Physical_Activity'].values.astype(float)
    alc   = df['Alcohol_Use'].values.astype(float)
    fhd   = df['Family_History_Diabetes'].values.astype(float)
    fhh   = df['Family_History_Heart'].values.astype(float)
    bps   = df['Blood_Pressure_Systolic'].values.astype(float)
    glc   = df['Glucose'].values.astype(float)
    hba1c = df['HbA1c'].values.astype(float)
    ldl   = df['LDL_Cholesterol'].values.astype(float)
    hdl   = df['HDL_Cholesterol'].values.astype(float)
    trig  = df['Triglycerides'].values.astype(float)
    cre   = df['Creatinine'].values.astype(float)
    alt   = df['ALT'].values.astype(float)
    ast   = df['AST'].values.astype(float)

    glc_safe = np.where(np.isnan(glc), 95.0, glc)

    # ── DIABETES ───────────────────────────────────────────────────────────────
    logit_d = (
        -9.0
        + 0.60 * hba1c
        + 0.012 * glc_safe
        + 0.05  * bmi
        + 0.025 * age
        + 0.70  * fhd
        + 0.35  * (bmi > 30).astype(float)
        - 0.25  * (pa >= 3).astype(float)
        + 0.20  * (pa == 1).astype(float)
    )
    prob_d = 1.0 / (1.0 + np.exp(-np.clip(logit_d, -15, 15)))
    diabetes = (rng.random(n) < prob_d).astype(int)

    # Hard overrides — clinical thresholds
    diabetes[hba1c >= 6.5] = 1                          # Diagnostic for diabetes
    diabetes[(hba1c < 5.0) & (glc_safe < 90)] = 0      # Definitively not diabetic
    diabetes[glc_safe >= 200] = 1                       # Severely elevated glucose

    # ── HEART DISEASE ──────────────────────────────────────────────────────────
    # Stronger intercept and coefficients so moderate-high values register clearly
    logit_h = (
        -8.0                                              # less conservative baseline
        + 0.050 * age                                     # age: stronger weight
        + 0.022 * bps                                     # systolic BP: stronger
        + 0.016 * ldl                                     # LDL: stronger
        - 0.025 * hdl                                     # HDL protective
        + 0.007 * trig                                    # triglycerides
        + 0.20  * cre                                     # creatinine
        + 1.20  * smk                                     # smoking: very strong
        + 0.80  * fhh                                     # family history: strong
        + 0.50  * (bps > 140).astype(float)               # hypertension flag: stronger
        + 0.40  * (ldl > 160).astype(float)               # high LDL flag: stronger
        + 0.30  * (hdl < 40).astype(float)                # low HDL flag
        + 0.25  * (diabetes == 1).astype(float)           # comorbidity
    )
    prob_h = 1.0 / (1.0 + np.exp(-np.clip(logit_h, -15, 15)))
    heart = (rng.random(n) < prob_h).astype(int)

    # Hard overrides — comprehensive clinical thresholds for Heart Disease
    heart[bps >= 180] = 1                                        # Stage 3 hypertension
    heart[(ldl >= 190) & (bps >= 145)] = 1                      # High LDL + elevated BP
    heart[(smk == 1) & (bps >= 155) & (age >= 48)] = 1          # Smoker + HTN + age
    heart[(smk == 1) & (fhh == 1) & (age >= 45)] = 1            # Smoker + family hx
    heart[(ldl >= 200) & (hdl <= 35)] = 1                       # Severe dyslipidemia
    heart[(fhh == 1) & (ldl >= 170) & (age >= 52)] = 1          # Family hx + high LDL
    heart[(bps >= 165) & (ldl >= 160) & (age >= 50)] = 1        # HTN + LDL + age
    heart[(smk == 1) & (ldl >= 180) & (hdl <= 40)] = 1          # Smoking + bad lipids
    # Definitively low risk
    heart[(age < 30) & (smk == 0) & (bps < 120) & (ldl < 100) & (fhh == 0)] = 0
    heart[(age < 25) & (smk == 0) & (bps < 115) & (ldl < 90)] = 0

    # ── LIVER DISEASE ──────────────────────────────────────────────────────────
    logit_l = (
        -7.0
        + 0.040 * alt
        + 0.032 * ast
        + 0.035 * bmi
        + 1.30  * (alc == 2).astype(float)
        + 0.60  * (alc == 1).astype(float)
        + 0.007 * trig
        + 0.30  * (bmi > 30).astype(float)
        + 0.20  * (diabetes == 1).astype(float)
        + 0.20  * smk
        - 0.25  * (pa >= 3).astype(float)
        + 0.020 * age
    )
    prob_l = 1.0 / (1.0 + np.exp(-np.clip(logit_l, -15, 15)))
    liver = (rng.random(n) < prob_l).astype(int)

    # Hard overrides — enzyme thresholds
    liver[alt >= 70]  = 1                               # ALT > 70 is clinically significant
    liver[ast >= 60]  = 1                               # AST > 60 is clinically significant
    liver[(alc == 2) & (bmi > 30)] = 1                 # Heavy drinker + obese = liver disease
    liver[(alt < 30) & (ast < 30) & (alc == 0) & (bmi < 26)] = 0  # Definitively healthy

    df = df.copy()
    df['Diabetes']     = diabetes
    df['Heart_Disease'] = heart
    df['Liver_Disease'] = liver
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  BASE REALISTIC POPULATION
# ─────────────────────────────────────────────────────────────────────────────
def generate_base(n: int = 8000, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age    = rng.integers(18, 86, n).astype(float)
    gender = rng.choice(['Male', 'Female'], n)
    male   = (gender == 'Male').astype(float)

    bmi    = np.clip(rng.normal(26.0, 5.5, n), 15.0, 55.0)
    smk_p  = np.clip(0.12 + 0.06 * male + 0.001 * np.clip(age - 40, 0, 20), 0, 0.5)
    smoking = (rng.random(n) < smk_p).astype(int)

    pa_p = np.clip(0.60 - 0.003 * age - 0.007 * bmi, 0.08, 0.88)
    pa   = np.where(rng.random(n) < pa_p,
                    rng.integers(3, 5, n),
                    rng.integers(1, 3, n)).astype(int)

    alc = rng.choice([0, 1, 2], n, p=[0.50, 0.38, 0.12])
    fhd = (rng.random(n) < 0.22).astype(int)
    fhh = (rng.random(n) < 0.20).astype(int)

    bps = np.clip(rng.normal(115 + 0.30 * age + 0.4 * bmi + 4 * smoking, 13, n), 85, 210)
    bpd = np.clip(rng.normal(68  + 0.18 * age + 0.28 * bmi + 3 * smoking, 9,  n), 50, 125)

    glc_base = 78 + 0.20 * age + 1.0 * bmi + 8 * fhd
    glc = np.clip(rng.normal(glc_base, 18, n), 55, 380).astype(float)
    glc[rng.random(n) < 0.04] = np.nan

    hba1c_base = 4.5 + (np.where(np.isnan(glc), 90, glc) / 100) * 1.6 + 0.008 * age
    hba1c = np.clip(rng.normal(hba1c_base, 0.65, n), 4.0, 14.0)

    chol = np.clip(rng.normal(165 + 0.7 * age + 2 * smoking + 0.4 * bmi, 35, n), 100, 400)
    hdl  = np.clip(rng.normal(60 - 7 * male - 4 * smoking - 0.15 * bmi, 12, n), 15, 100)
    ldl  = np.clip(rng.normal((chol - hdl) * 0.62, 26, n), 25, 270)
    trig = np.clip(rng.normal(90 + 2.5 * bmi + 15 * alc + 0.25 * age, 42, n), 35, 600)
    cre  = np.clip(rng.normal(0.7 + 0.28 * male + 0.004 * age + 0.0008 * bps, 0.3, n), 0.3, 14.0)

    # Base population has mostly NORMAL liver values
    alt = np.clip(rng.normal(20 + 0.4 * bmi + 5 * alc + 0.05 * age, 12, n), 5, 300)
    ast = np.clip(rng.normal(alt * 0.70 + 3 * smoking, 10, n), 5, 250)

    df = pd.DataFrame({
        'Age': age, 'Gender': gender, 'BMI': bmi.round(1),
        'Blood_Pressure_Systolic': bps.round(1), 'Blood_Pressure_Diastolic': bpd.round(1),
        'Glucose': glc.round(1), 'HbA1c': hba1c.round(2),
        'Cholesterol': chol.round(1), 'HDL_Cholesterol': hdl.round(1),
        'LDL_Cholesterol': ldl.round(1), 'Triglycerides': trig.round(1),
        'Creatinine': cre.round(2), 'ALT': alt.round(1), 'AST': ast.round(1),
        'Smoking': smoking, 'Physical_Activity': pa, 'Alcohol_Use': alc,
        'Family_History_Diabetes': fhd, 'Family_History_Heart': fhh,
    })
    return assign_labels(df, rng)


# ─────────────────────────────────────────────────────────────────────────────
#  TARGETED EXTREME / BORDERLINE / HEALTHY CASES
# ─────────────────────────────────────────────────────────────────────────────
def generate_targeted(seed: int = 9999) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    def R(lo, hi):  return round(float(rng.uniform(lo, hi)), 1)
    def Ri(lo, hi): return int(rng.integers(lo, hi + 1))
    def Rc(lo, hi): return round(float(rng.uniform(lo, hi)), 2)
    def G():        return rng.choice(['Male', 'Female'])

    # ──────────────────────────────────────────────────────────────────────────
    # A. LOW-RISK HEALTHY PATIENTS  (no disease)
    # ──────────────────────────────────────────────────────────────────────────
    for _ in range(1500):
        rows.append({
            'Age': Ri(18, 45), 'Gender': G(), 'BMI': R(17.5, 24.5),
            'Blood_Pressure_Systolic': R(90, 118), 'Blood_Pressure_Diastolic': R(58, 78),
            'Glucose': R(68, 99), 'HbA1c': Rc(4.0, 5.4),
            'Cholesterol': R(130, 195), 'HDL_Cholesterol': R(55, 100),
            'LDL_Cholesterol': R(48, 99), 'Triglycerides': R(48, 145),
            'Creatinine': Rc(0.5, 1.1), 'ALT': R(7, 28), 'AST': R(7, 25),
            'Smoking': 0, 'Physical_Activity': Ri(3, 4), 'Alcohol_Use': 0,
            'Family_History_Diabetes': 0, 'Family_History_Heart': 0,
        })

    # ──────────────────────────────────────────────────────────────────────────
    # B. HIGH DIABETES RISK (HbA1c high + Glucose high)
    # ──────────────────────────────────────────────────────────────────────────
    for _ in range(800):
        rows.append({
            'Age': Ri(40, 82), 'Gender': G(), 'BMI': R(30, 55),
            'Blood_Pressure_Systolic': R(120, 170), 'Blood_Pressure_Diastolic': R(75, 112),
            'Glucose': R(180, 380), 'HbA1c': Rc(8.0, 14.0),
            'Cholesterol': R(180, 280), 'HDL_Cholesterol': R(22, 52),
            'LDL_Cholesterol': R(110, 220), 'Triglycerides': R(140, 420),
            'Creatinine': Rc(0.7, 2.5), 'ALT': R(20, 75), 'AST': R(18, 70),
            'Smoking': Ri(0, 1), 'Physical_Activity': Ri(1, 2), 'Alcohol_Use': Ri(0, 1),
            'Family_History_Diabetes': 1, 'Family_History_Heart': Ri(0, 1),
        })

    # MEDIUM DIABETES RISK (HbA1c 6.5-8.0)
    for _ in range(600):
        rows.append({
            'Age': Ri(35, 70), 'Gender': G(), 'BMI': R(27, 38),
            'Blood_Pressure_Systolic': R(115, 148), 'Blood_Pressure_Diastolic': R(70, 96),
            'Glucose': R(130, 200), 'HbA1c': Rc(6.5, 8.0),
            'Cholesterol': R(195, 255), 'HDL_Cholesterol': R(32, 55),
            'LDL_Cholesterol': R(125, 185), 'Triglycerides': R(148, 310),
            'Creatinine': Rc(0.7, 1.6), 'ALT': R(18, 55), 'AST': R(15, 48),
            'Smoking': Ri(0, 1), 'Physical_Activity': Ri(2, 3), 'Alcohol_Use': Ri(0, 1),
            'Family_History_Diabetes': Ri(0, 1), 'Family_History_Heart': Ri(0, 1),
        })

    # ──────────────────────────────────────────────────────────────────────────
    # C. HIGH HEART DISEASE RISK
    # ──────────────────────────────────────────────────────────────────────────
    for _ in range(1200):   # increased from 800 to 1200
        rows.append({
            'Age': Ri(55, 85), 'Gender': rng.choice(['Male', 'Male', 'Female']),
            'BMI': R(25, 45),
            'Blood_Pressure_Systolic': R(162, 220), 'Blood_Pressure_Diastolic': R(95, 132),
            'Glucose': R(88, 145), 'HbA1c': Rc(5.0, 7.2),
            'Cholesterol': R(255, 400), 'HDL_Cholesterol': R(15, 38),
            'LDL_Cholesterol': R(175, 285), 'Triglycerides': R(195, 510),
            'Creatinine': Rc(1.0, 3.8), 'ALT': R(15, 52), 'AST': R(15, 48),
            'Smoking': 1, 'Physical_Activity': 1, 'Alcohol_Use': Ri(0, 1),
            'Family_History_Diabetes': Ri(0, 1), 'Family_History_Heart': 1,
        })

    # MODERATE-HIGH HEART RISK (BP 155-180, LDL 165-220, smoker/family hx)
    # This is the critical gap — values users type as 'large' but model misses
    for _ in range(700):
        rows.append({
            'Age': Ri(48, 72), 'Gender': rng.choice(['Male', 'Male', 'Female']),
            'BMI': R(26, 40),
            'Blood_Pressure_Systolic': R(155, 182), 'Blood_Pressure_Diastolic': R(92, 115),
            'Glucose': R(88, 145), 'HbA1c': Rc(5.0, 7.2),
            'Cholesterol': R(235, 330), 'HDL_Cholesterol': R(22, 42),
            'LDL_Cholesterol': R(165, 225), 'Triglycerides': R(165, 420),
            'Creatinine': Rc(0.9, 2.2), 'ALT': R(18, 52), 'AST': R(15, 48),
            'Smoking': 1, 'Physical_Activity': Ri(1, 2), 'Alcohol_Use': Ri(0, 1),
            'Family_History_Diabetes': Ri(0, 1), 'Family_History_Heart': 1,
        })

    # MEDIUM HEART DISEASE RISK
    for _ in range(500):
        rows.append({
            'Age': Ri(45, 68), 'Gender': G(), 'BMI': R(27, 38),
            'Blood_Pressure_Systolic': R(142, 165), 'Blood_Pressure_Diastolic': R(87, 108),
            'Glucose': R(88, 145), 'HbA1c': Rc(5.0, 7.2),
            'Cholesterol': R(218, 280), 'HDL_Cholesterol': R(28, 48),
            'LDL_Cholesterol': R(148, 200), 'Triglycerides': R(148, 310),
            'Creatinine': Rc(0.8, 1.9), 'ALT': R(18, 55), 'AST': R(15, 50),
            'Smoking': Ri(0, 1), 'Physical_Activity': Ri(1, 2), 'Alcohol_Use': Ri(0, 1),
            'Family_History_Diabetes': Ri(0, 1), 'Family_History_Heart': 1,
        })

    # ──────────────────────────────────────────────────────────────────────────
    # D. HIGH LIVER DISEASE RISK (very elevated ALT/AST + alcohol/obesity)
    # ──────────────────────────────────────────────────────────────────────────
    for _ in range(1000):
        rows.append({
            'Age': Ri(30, 75), 'Gender': G(), 'BMI': R(30, 58),
            'Blood_Pressure_Systolic': R(110, 162), 'Blood_Pressure_Diastolic': R(68, 102),
            'Glucose': R(88, 175), 'HbA1c': Rc(4.8, 8.0),
            'Cholesterol': R(175, 268), 'HDL_Cholesterol': R(22, 52),
            'LDL_Cholesterol': R(98, 188), 'Triglycerides': R(175, 520),
            'Creatinine': Rc(0.7, 1.9), 'ALT': R(90, 500), 'AST': R(80, 460),
            'Smoking': Ri(0, 1), 'Physical_Activity': Ri(1, 2), 'Alcohol_Use': 2,
            'Family_History_Diabetes': Ri(0, 1), 'Family_History_Heart': Ri(0, 1),
        })

    # MEDIUM LIVER DISEASE RISK (elevated but not extreme enzymes)
    for _ in range(800):
        rows.append({
            'Age': Ri(28, 68), 'Gender': G(), 'BMI': R(28, 48),
            'Blood_Pressure_Systolic': R(108, 150), 'Blood_Pressure_Diastolic': R(68, 96),
            'Glucose': R(88, 158), 'HbA1c': Rc(4.8, 7.5),
            'Cholesterol': R(175, 252), 'HDL_Cholesterol': R(28, 55),
            'LDL_Cholesterol': R(98, 170), 'Triglycerides': R(148, 368),
            'Creatinine': Rc(0.7, 1.6), 'ALT': R(55, 130), 'AST': R(50, 120),
            'Smoking': Ri(0, 1), 'Physical_Activity': Ri(1, 2), 'Alcohol_Use': Ri(1, 2),
            'Family_History_Diabetes': Ri(0, 1), 'Family_History_Heart': Ri(0, 1),
        })

    # ──────────────────────────────────────────────────────────────────────────
    # E. ALL THREE DISEASES SIMULTANEOUSLY (multi-morbidity)
    # ──────────────────────────────────────────────────────────────────────────
    for _ in range(500):
        rows.append({
            'Age': Ri(52, 82), 'Gender': G(), 'BMI': R(34, 58),
            'Blood_Pressure_Systolic': R(158, 220), 'Blood_Pressure_Diastolic': R(92, 135),
            'Glucose': R(185, 380), 'HbA1c': Rc(8.0, 14.0),
            'Cholesterol': R(245, 410), 'HDL_Cholesterol': R(15, 35),
            'LDL_Cholesterol': R(175, 290), 'Triglycerides': R(195, 540),
            'Creatinine': Rc(1.2, 4.2), 'ALT': R(95, 460), 'AST': R(85, 430),
            'Smoking': 1, 'Physical_Activity': 1, 'Alcohol_Use': 2,
            'Family_History_Diabetes': 1, 'Family_History_Heart': 1,
        })

    # ──────────────────────────────────────────────────────────────────────────
    # F. MIXED MODERATE-RISK PATIENTS
    # ──────────────────────────────────────────────────────────────────────────
    for _ in range(500):
        rows.append({
            'Age': Ri(45, 70), 'Gender': G(), 'BMI': R(28, 38),
            'Blood_Pressure_Systolic': R(130, 155), 'Blood_Pressure_Diastolic': R(82, 100),
            'Glucose': R(115, 175), 'HbA1c': Rc(6.0, 8.0),
            'Cholesterol': R(210, 265), 'HDL_Cholesterol': R(35, 55),
            'LDL_Cholesterol': R(140, 195), 'Triglycerides': R(145, 290),
            'Creatinine': Rc(0.9, 1.7), 'ALT': R(55, 120), 'AST': R(50, 110),
            'Smoking': Ri(0, 1), 'Physical_Activity': Ri(1, 3), 'Alcohol_Use': Ri(0, 2),
            'Family_History_Diabetes': Ri(0, 1), 'Family_History_Heart': Ri(0, 1),
        })

    df = pd.DataFrame(rows)
    return assign_labels(df, rng)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def generate_data():
    print("[1/3] Generating base population (8,000 patients)...")
    base_df = generate_base(8000)

    print("[2/3] Generating targeted extreme/borderline cases (7,000 patients)...")
    extra_df = generate_targeted()

    print("[3/3] Combining, shuffling and saving...")
    combined = pd.concat([base_df, extra_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=SEED).reset_index(drop=True)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'patient_records.csv')
    combined.to_csv(out_path, index=False)

    total = len(combined)
    print(f"\n[OK] Generated {total} records -> {out_path}")
    print("\n[Class Balance]")
    for col in ['Diabetes', 'Heart_Disease', 'Liver_Disease']:
        pos = combined[col].sum()
        pct = combined[col].mean() * 100
        print(f"  {col:<20}: {pos:>5} positive ({pct:.1f}%)")
    return combined


if __name__ == "__main__":
    generate_data()
