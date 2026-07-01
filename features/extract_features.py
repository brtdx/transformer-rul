#!/usr/bin/env python3
"""Cycle-level feature extraction from 4 channels for RUL Transformer.

Features per cycle (~14):
  From cycle table: ChgCap, DChgCap, Eff, ChgEnergy, DChgEnergy, ChgTime, DChgTime
                    Ch3: Ah_Throughput_DChg, Ah_Throughput_Chg
  From step table:
    CV_Chg_Time       SUM StepTime WHERE StepType=2 GROUP BY CycleIndex
    IR_drop_ohm       (rest_end_V - dchg_oneset_V) / |I_nominal|  (Rest->CC_DChg)
    Relax_rate_V_s    (rest_end_V - dchg_end_V)/rest_time          (CC_DChg->Rest)
    DChg_end_V        last discharge voltage (low-voltage cutoff trend)

Output: /home/bbb/Desktop/rul/features/features_all_cells.hdf5
  /cell_1, /cell_2, /cell_3, /cell_6  datasets:
    X         (N_cycles, F) float32   features (cycle 0 formation DROPped)
    SoH       (N_cycles,)   float32    DChgCap / peak_DChgCap  (relative SoH %)
    CycleID   (N_cycles,)   int32       1..N
"""
import duckdb
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

CELLS = [
    # name, duckdb_path, nominal_dchg_current_A, qnom_ch1_dchg_A_for_relative
    ('cell_1', '/home/bbb/Desktop/cell1/Channel_1.duckdb', 6.0,  None),
    ('cell_2', '/home/bbb/Desktop/cell1/cell2/Channel_2.duckdb', 12.0, None),
    ('cell_3', '/home/bbb/Desktop/cell1/cell3/Channel_3.duckdb',  None, None),
    ('cell_6', '/home/bbb/Desktop/cell1/cell6/Channel_6.duckdb', 12.0, None),
]

OUT = Path('/home/bbb/Desktop/rul/features/features_all_cells.hdf5')


def load_cycle(name, db_path):
    """Pull raw cycle table (DuckDB), unordered until sorted downstream."""
    con = duckdb.connect(db_path, read_only=True)
    cols = ['"Cycle Index"', '"Chg. Cap.(Ah)"', '"DChg. Cap.(Ah)"',
            '"Chg.-DChg. Eff(%)"', '"Chg. Energy(Wh)"', '"DChg. Energy(Wh)"',
            '"Chg. Time"', '"DChg. Time"']
    sel = ', '.join(f'{c} as "{c.strip(chr(34))}"' for c in cols)
    df = con.execute(f'SELECT {sel} FROM "cycle" ORDER BY "Cycle Index"').fetchdf()
    df.columns = ['CycleIndex', 'ChgCap', 'DChgCap', 'Eff_percent',
                  'ChgEnergy', 'DChgEnergy', 'ChgTime', 'DChgTime']
    # ChgTime/DChgTime may be VARCHAR 'HH:MM:SS' (Ch3) or numeric seconds (Ch1/Ch2/Ch6)
    def to_sec(s):
        if isinstance(s, str):
            if 'days' in s:
                return float(pd.to_timedelta(s).total_seconds())
            parts = s.split(':')
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            return float(s)
        return float(s)
    df['ChgTime'] = df['ChgTime'].apply(to_sec)
    df['DChgTime'] = df['DChgTime'].apply(to_sec)
    # Ch3 extras (best-effort, ignore if absent)
    for col, alias in [('Ah_Throughput_DChg', 'AhThroughputDChg'),
                       ('Ah_Throughput_Chg', 'AhThroughputChg')]:
        try:
            df[alias] = con.execute(f'SELECT "{col}" FROM "cycle" ORDER BY "Cycle Index"').fetchdf().iloc[:, 0]
        except Exception:
            df[alias] = np.nan
    con.close()
    return df


def step_features(name, db_path, nominal_dchg_A):
    """Pull standard + IR drop + relaxation features from step table.

    Handles two schemas:
      Ch1/Ch2/Ch6: StepType TINYINT (0,1,2,3)
      Ch3:         StepType VARCHAR ("Rest","CC Chg","CC DChg","CV Chg")
    """
    con = duckdb.connect(db_path, read_only=True)
    # detect schema
    st_type = con.execute(
        "SELECT data_type FROM duckdb_columns() WHERE table_name='step' AND column_name='Step Type'"
    ).fetchone()[0]
    is_varchar = 'VARCHAR' in st_type
    # Normalize step: keep numeric StepType_tinyint + StepTime_s + Date->unix if needed
    if is_varchar:
        st_map_sql = """
          CASE "Step Type"
            WHEN 'Rest' THEN 0
            WHEN 'CC Chg' THEN 1
            WHEN 'CV Chg' THEN 2
            WHEN 'CC DChg' THEN 3
            ELSE -1 END
        """
        time_col = "TRY_CAST(strptime(replace(\"Step Time\",' days ',' '),'%H:%M:%S') AS TIME)"  # sad path; use fallback below
        # Ch3 step Time format 'HH:MM:SS' string -> seconds
        def parse_hhmmss(s):
            if s is None or len(s) == 0:
                return np.nan
            parts = s.split(':')
            if len(parts) != 3:
                return np.nan
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        # Pull StepType_str + raw time + voltages
        df = con.execute("""
          SELECT "Cycle Index" as CycleIndex, "Step Index", "Step Number",
                 "Step Type", "Step Time",
                 "Oneset Volt.(V)" as OnesetV, "End Voltage(V)" as EndV
          FROM "step" ORDER BY "Step Number"
        """).fetchdf()
        df['StepType'] = df['Step Type'].map({'Rest': 0, 'CC Chg': 1, 'CV Chg': 2, 'CC DChg': 3}).fillna(-1).astype(int)
        # Step Time 'HH:MM:SS' -> seconds  (rare "N days HH:MM:SS" handled gracefully)
        def parse(s):
            if not isinstance(s, str):
                return np.nan
            if 'days' in s:
                return float(pd.to_timedelta(s).total_seconds())
            parts = s.split(':')
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        df['StepTime_s'] = df['Step Time'].apply(parse)
    else:
        df = con.execute("""
          SELECT "Cycle Index" as CycleIndex, "Step Index", "Step Number",
                 "Step Type" as StepType, "Step Time" as StepTime_s,
                 "Oneset Volt.(V)" as OnesetV, "End Voltage(V)" as EndV
          FROM "step" ORDER BY "Step Number"
        """).fetchdf()
    con.close()

    cv_times = (df[df['StepType'] == 2]
                .groupby('CycleIndex')['StepTime_s'].sum().rename('CV_Chg_Time'))

    # IR drop: find DChg steps preceded by Rest (LAG by Step Number within overall order)
    df_sorted = df.sort_values('Step Number').reset_index(drop=True).copy()
    df_sorted['prev_st'] = df_sorted['StepType'].shift(1)
    df_sorted['prev_end_v'] = df_sorted['EndV'].shift(1)
    df_sorted['next_st'] = df_sorted['StepType'].shift(-1)

    dchg_starts = df_sorted[(df_sorted['StepType'] == 3) & (df_sorted['prev_st'] == 0)].copy()
    # IR_drop_dV = prev_rest_end_v - dchg_oneset_v   (always >= 0)
    dchg_starts['IR_drop_dV'] = dchg_starts['prev_end_v'] - dchg_starts['OnesetV']
    # Normalize by nominal discharge current. For Ch3 DST we use the average current
    # of the first 1s of the discharge step's parent cycle (computed downstream). For now,
    # store raw delta-V; a separate normalized column will be filled for Std cells.
    if nominal_dchg_A is not None:
        dchg_starts['IR_drop_ohm'] = dchg_starts['IR_drop_dV'] / abs(nominal_dchg_A)
    else:
        dchg_starts['IR_drop_ohm'] = np.nan
    ir = (dchg_starts[['CycleIndex', 'IR_drop_dV', 'IR_drop_ohm']]
          .groupby('CycleIndex').mean())

    # Relaxation: rest steps preceded by DChg
    rests = df_sorted[(df_sorted['StepType'] == 0) & (df_sorted['prev_st'] == 3)].copy()
    # We need dchg_end_v immediately before this rest -> prev_end_v (it IS the dchg's end_v)
    rests['relax_dV'] = rests['EndV'] - rests['prev_end_v']
    rests['relax_rate_V_s'] = rests['relax_dV'] / rests['StepTime_s']
    relax = (rests[['CycleIndex', 'relax_dV', 'relax_rate_V_s']]
             .groupby('CycleIndex').mean())

    # DChg end voltage per cycle (low-voltage cutoff trend)
    dchg_rows = df_sorted[df_sorted['StepType'] == 3]
    dchg_end = dchg_rows.groupby('CycleIndex')['EndV'].mean().rename('DChg_end_V')

    feats = pd.concat([cv_times, ir, relax, dchg_end], axis=1)
    return feats


def ch3_dchg_current_avg(cycle_idx):
    """Placeholder: for Ch3 we still leave IR_drop_ohm as NaN; we use IR_drop_dV only.

    (DST discharge current is variable, so resistance in ohm is ill-defined per-cycle.
     Cycling ΔV_dischargestart is sufficient signal.)
    """
    return np.nan


def build_cell(name, db_path, nominal_dchg_A):
    cyc_df = load_cycle(name, db_path)
    feats = step_features(name, db_path, nominal_dchg_A)
    # Join on CycleIndex
    merged = cyc_df.merge(feats, left_on='CycleIndex', right_index=True, how='left')
    # Drop cycle 0 (formation), and any rows with CycleIndex < 1
    merged = merged[merged['CycleIndex'] >= 1].reset_index(drop=True)

    # Feature columns (ordered, fixed across cells; missing cols get NaN)
    feat_cols = ['ChgCap', 'DChgCap', 'Eff_percent', 'ChgEnergy', 'DChgEnergy',
                 'ChgTime', 'DChgTime', 'CV_Chg_Time', 'IR_drop_dV', 'IR_drop_ohm',
                 'relax_dV', 'relax_rate_V_s', 'DChg_end_V',
                 'AhThroughputDChg', 'AhThroughputChg']
    for c in feat_cols:
        if c not in merged.columns:
            merged[c] = np.nan
    # Clean noisy Cycles: DChgCap or ChgCap implausibly low (<10% of peak) -> NaN + linear interp
    # (fixes Ch2 cycles 347-348 NEWARE logging anomalies)
    peak_dchg = merged['DChgCap'].max()
    peak_chg = merged['ChgCap'].max()
    bad_mask = (merged['DChgCap'] < 0.10 * peak_dchg) | (merged['ChgCap'] < 0.10 * peak_chg)
    n_bad = int(bad_mask.sum())
    if n_bad > 0:
        merged.loc[bad_mask, ['DChgCap', 'ChgCap']] = np.nan
        merged['DChgCap'] = merged['DChgCap'].interpolate(limit_direction='both')
        merged['ChgCap'] = merged['ChgCap'].interpolate(limit_direction='both')
        # Also fix Eff when either was missing (avoid Eff=0% from NEWARE artifact)
        bad_eff = (merged['Eff_percent'] < 50.0) | (merged['Eff_percent'] > 101.0)
        if bad_eff.any():
            merged.loc[bad_eff, 'Eff_percent'] = np.nan
            merged['Eff_percent'] = merged['Eff_percent'].interpolate(limit_direction='both')
        print(f"    [clean] {name}: interpolated {n_bad} anomalous DChgCap/ChgCap cycles")

    X = merged[feat_cols].astype(np.float32).to_numpy()
    SoH = (merged['DChgCap'] / merged['DChgCap'].max() * 100.0).astype(np.float32).to_numpy()
    CycleID = merged['CycleIndex'].astype(np.int32).to_numpy()
    return X, SoH, CycleID, feat_cols


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    feature_names = None
    summary = {}
    with h5py.File(OUT, 'w') as h5:
        for name, db_path, dchg_A, _ in CELLS:
            print(f"  Building {name} from {db_path}")
            X, SoH, CycleID, feat_cols = build_cell(name, db_path, dchg_A)
            if feature_names is None:
                feature_names = feat_cols
            # assert same feature set across cells
            assert feat_cols == feature_names, f"{name} feature mismatch"
            g = h5.create_group(name)
            g.create_dataset('X', data=X, chunks=(min(64, X.shape[0]), X.shape[1]))
            g.create_dataset('SoH', data=SoH)
            g.create_dataset('CycleID', data=CycleID)
            g.attrs['n_cycles'] = X.shape[0]
            g.attrs['peak_dchgcap_Ah'] = float((X[:, list(feature_names).index('DChgCap')]).max())
            summary[name] = (X.shape[0], X.shape[1], float(SoH[0]), float(SoH[-1]),
                             float(SoH.min()))
        # Store global feature names as attribute on root
        h5.attrs['feature_names'] = np.array(feature_names, dtype='S')
        h5.attrs['horizons'] = np.array([5, 10, 20], dtype=np.int32)

    print("\n=== Feature extraction summary ===")
    print(f"  Features: {feature_names}")
    for name, (n, f, s0, sf, smin) in summary.items():
        print(f"  {name}: {n} cycles, {f} feats; SoH {s0:.2f}% -> {sf:.2f}% (min={smin:.2f}%)")
    print(f"\nOutput: {OUT}  ({OUT.stat().st_size/1024:.1f} KB)")


if __name__ == '__main__':
    main()