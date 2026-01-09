# Feature Extraction Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    extract_features_for_task()                               │
│                    (Main entry point - features.py:325)                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. FeatureExtractor.__init__()  (line 75)                                   │
│    └─ Creates ClifOrchestrator with config_path                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. load_tables()  (line 86)                                                 │
│    Loads 5 CLIF tables filtered by hospitalization_id:                      │
│    ├─ vitals        → heart_rate, map, sbp, respiratory_rate, spo2, temp_c  │
│    ├─ labs          → albumin, creatinine, lactate, platelets, etc (17)     │
│    ├─ respiratory_support → device_category, fio2_set, peep_set             │
│    ├─ medication_admin_continuous → vasopressors (8 types)                  │
│    └─ patient_assessments → gcs_total                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. apply_outlier_handling()  (line 137)                                     │
│    └─ Uses clifpy.utils.outlier_handler.apply_outlier_handling()            │
│       Nullifies physiologically implausible values                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. create_wide_dataset()  (line 149)                                        │
│    └─ clif.create_wide_dataset()                                            │
│       - Filters by time window (window_start → window_end)                  │
│       - Pivots tables: each category becomes a column                       │
│       - Creates event-level rows (many rows per hospitalization)            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. aggregate_features()  (line 190)                                         │
│    Collapses events → ONE ROW per hospitalization:                          │
│                                                                              │
│    ├─ MAX: lactate, bun, creatinine, ast, alt, bilirubin, inr, ptt, pt,    │
│    │       fio2_set, peep_set, resp_rate, heart_rate, temp_c, sodium, etc. │
│    │                                                                         │
│    ├─ MIN: platelet_count, po2_arterial, spo2, sbp, map, albumin,          │
│    │       sodium, potassium, wbc, temp_c, heart_rate                       │
│    │                                                                         │
│    ├─ MEDIAN: respiratory_rate, fio2_set, bilirubin_total, map             │
│    │                                                                         │
│    └─ LAST: gcs_total (most recent value)                                   │
│                                                                              │
│    Then adds derived features:                                               │
│    ├─ _add_device_encoding() → One-hot: imv, nippv, cpap, high_flow_nc,    │
│    │                           face_mask, trach_collar, nasal_cannula       │
│    └─ _add_vasopressor_count() → Count of unique vasopressors used          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 6. add_demographics()  (line 288)                                           │
│    Merges from cohort:                                                       │
│    ├─ age_at_admission → age                                                │
│    ├─ sex_category → isfemale (binary)                                      │
│    ├─ race_category                                                          │
│    └─ ethnicity_category                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                         Final features DataFrame
                      (1 row per hospitalization)
```

## Key Functions Summary

| Function | Location | Purpose |
|----------|----------|---------|
| `extract_features_for_task()` | features.py:325 | Main entry point, orchestrates everything |
| `FeatureExtractor.load_tables()` | features.py:86 | Loads CLIF tables with filters |
| `apply_outlier_handling()` | features.py:137 | Removes physiologically invalid values |
| `create_wide_dataset()` | features.py:149 | Creates event-level wide format |
| `aggregate_features()` | features.py:190 | Aggregates to 1 row per patient |
| `_add_device_encoding()` | features.py:248 | One-hot encodes respiratory devices |
| `_add_vasopressor_count()` | features.py:270 | Counts vasopressor types used |
| `add_demographics()` | features.py:288 | Adds age, sex, race, ethnicity |

## Feature Categories

### Vitals (6)
- heart_rate, map, sbp, respiratory_rate, spo2, temp_c

### Labs (17)
- albumin, alt, ast, bicarbonate, bilirubin_total, bun, chloride, creatinine
- inr, lactate, platelet_count, po2_arterial, potassium, pt, ptt, sodium, wbc

### Respiratory Support (3 + 7 devices)
- fio2_set, peep_set, device_category
- Devices one-hot: imv, nippv, cpap, high_flow_nc, face_mask, trach_collar, nasal_cannula

### Medications (8 vasopressors)
- norepinephrine, epinephrine, phenylephrine, vasopressin
- dopamine, dobutamine, milrinone, isoproterenol
- Derived: vasopressor_count

### Patient Assessments (1)
- gcs_total

### Demographics (4)
- age, isfemale, race_category, ethnicity_category
