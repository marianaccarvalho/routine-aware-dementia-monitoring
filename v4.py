"""
=============================================================================
  PIPELINE DE SIMULAÇÃO — Anomaly Detection on Daily Routines
  VERSION: 11  (revisão SMHL-D-26-00855)

  Base: simulation_pipeline_v10.py (resultados do paper submetido).

  Alterações v11 (pedidos do Reviewer 2):
    R2.4  PARTICIPANT_SPLIT — rotinas de treino/calibração geradas só com
          segmentos de participantes de treino; rotinas de avaliação só com
          participantes de teste (participant-independent split).
    R2.6  RUN_ABLATION — todas as variantes avaliadas nos MESMOS pares de
          avaliação: só LSTM, só duração, sem cascade, k de suavização,
          percentil do limiar, limiar global vs por classe, calibração
          treino (P95) vs held-out.
    R2.5  RUN_SEVERITY — anomalias de duração com severidade graduada,
          incluindo casos DENTRO dos limites [0.3×dmin, 3.0×dmax].
    R2.3  RUN_PERTURB — rotinas normais perturbadas (passo saltado, passos
          trocados, interrupção, jitter de duração) → FAR.
    Outros
      PAD_MODE='edge' — o contexto das primeiras 47 janelas é preenchido com
          a primeira atividade observada (v10: índice 0 = 'Eating with
          cutlery'). PAD_MODE='zero' reproduz a v10.
      DUR_SIGNAL — sinal usado para segmentar blocos no detetor de duração.
          v10 usava 'activity_true' (ground truth); o paper descreve o fluxo
          suavizado → 'activity_smoothed'. Default mantém a v10.
      Classificação em batch (uma chamada predict por rotina) → mais rápido,
          mesmos resultados.
=============================================================================
"""

import os, re, random, warnings
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, RepeatVector, TimeDistributed, Embedding
)
from tensorflow.keras.callbacks import EarlyStopping
tf.random.set_seed(42)

VERSION = 11
print(f"\n{'='*60}")
print(f"  PIPELINE DE SIMULAÇÃO  v{VERSION}")
print(f"{'='*60}\n")

# =============================================================================
# 0. CONFIGURAÇÃO
# =============================================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
_local_data = os.path.join(BASE_DIR, "Data")
_har_data   = os.path.join(os.path.dirname(BASE_DIR), "HAR_nossos dados", "Data")
DATA_DIR   = os.environ.get("PIPELINE_DATA_DIR") or (
             _local_data if os.path.isdir(_local_data) else _har_data)
OUTPUT_DIR = os.path.join(BASE_DIR, "output_v11")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED          = 42
DUR_MARGIN_LO        = 0.30
DUR_MARGIN_HI        = 3.00
SMOOTH_K             = 5
WINDOW_SIZE          = 60
STEP_SIZE            = 30
N_TRAIN_ROUTINES     = int(os.environ.get("PIPELINE_N_TRAIN", 150))
BEHAVIORAL_SIGNAL    = 'activity_smoothed'
CASCADE_SUPPRESS     = True

# ── LSTM Autoencoder ──────────────────────────────────────────────────────────
LSTM_SEQ_LEN     = 48
LSTM_EMBED_DIM   = 8
LSTM_UNITS       = 64
LSTM_EPOCHS      = int(os.environ.get("PIPELINE_EPOCHS", 50))
LSTM_BATCH       = 64
LSTM_AE_THRESH_P = 97
N_CALIB          = 10
N_EVAL_ROUTINES  = int(os.environ.get("PIPELINE_N_EVAL", 20))

# ── Novas opções v11 ──────────────────────────────────────────────────────────
PARTICIPANT_SPLIT    = True      # R2.4
N_TEST_PARTICIPANTS  = 3         # nº de participantes reservados para teste
TEST_PARTICIPANTS    = None      # ex.: ['03','07','11'] para fixar; None = automático
PAD_MODE             = 'edge'    # 'edge' (v11) | 'zero' (v10)
DUR_SIGNAL           = 'activity_true'   # 'activity_true' (v10) | 'activity_smoothed'

RUN_ABLATION  = True             # R2.6
RUN_SEVERITY  = os.environ.get("PIPELINE_SEVERITY", "1") == "1"   # R2.5
RUN_PERTURB   = os.environ.get("PIPELINE_PERTURB", "1") == "1"    # R2.3
N_SEVERITY    = int(os.environ.get("PIPELINE_N_SEV", 10))      # rotinas por fator
N_PERTURB     = int(os.environ.get("PIPELINE_N_PERTURB", 10))  # rotinas por tipo
SEV_LONG_FACTORS  = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]  # × dmax  (limite = 3.0)
SEV_SHORT_FACTORS = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]           # × dmin  (limite = 0.3)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print(f"Dados              : {DATA_DIR}")
print(f"Saída              : {OUTPUT_DIR}")
print(f"N_TRAIN_ROUTINES   = {N_TRAIN_ROUTINES}  | N_EVAL = {N_EVAL_ROUTINES}")
print(f"Participant split  = {PARTICIPANT_SPLIT}  | PAD_MODE = {PAD_MODE}  | DUR_SIGNAL = {DUR_SIGNAL}")
print(f"Ablação={RUN_ABLATION}  Severidade={RUN_SEVERITY}  Perturbações={RUN_PERTURB}\n")

# =============================================================================
# 1. CARREGAMENTO DOS DADOS
# =============================================================================
print("─── 1. Carregamento dos dados ───────────────────────────────────────")

def robust_read_csv(path):
    for sep in [',', ';', '\t']:
        try:
            df = pd.read_csv(path, sep=sep, engine='python', on_bad_lines='skip')
            if df.shape[0] > 0 and df.shape[1] >= 2:
                return df
        except Exception:
            pass
    return pd.read_csv(path, engine='python', on_bad_lines='skip')

def parse_folder_name(name):
    parts = name.split('_')
    if len(parts) < 3 or parts[0].lower() != 'id':
        return 'unknown', 'unknown'
    pid, tokens = parts[1], []
    for t in parts[2:]:
        if t.isdigit() or re.match(r'^\d+s?$', t): break
        tokens.append(t)
    return pid, ' '.join(tokens) if tokens else 'unknown'

def read_sc(path):
    try:
        df = pd.read_csv(path, comment='#', header=None,
                         engine='python', on_bad_lines='skip')
        if df.shape[1] >= 3:
            df.columns = ['timestamp','spo2','hr'] + [f'x{i}' for i in range(df.shape[1]-3)]
        elif df.shape[1] == 2:
            df.columns = ['spo2','hr']
            df.insert(0,'timestamp', range(len(df)))
        else:
            return pd.DataFrame(columns=['timestamp','spo2','hr'])
        df['spo2'] = pd.to_numeric(df['spo2'], errors='coerce')
        df['hr']   = pd.to_numeric(df['hr'],   errors='coerce')
        df = df[['timestamp','spo2','hr']].dropna()
        return df[(df['spo2']>0) & (df['hr']>0)].reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=['timestamp','spo2','hr'])

def read_wearable(path):
    df = robust_read_csv(path)
    skip = ('atividade','activity','label','time','timestamp',
            'tempo_s','hr_bpm','spo2_percent')
    cols = [c for c in df.columns if c.lower() not in skip]
    df = df[cols].select_dtypes(include=[np.number])
    return df.replace([np.inf,-np.inf], np.nan).dropna(how='all').reset_index(drop=True)

segments, skipped = [], []
for folder in sorted(d for d in os.listdir(DATA_DIR)
                     if os.path.isdir(os.path.join(DATA_DIR,d)) and d.startswith('ID_')):
    fpath = os.path.join(DATA_DIR, folder)
    pid, act = parse_folder_name(folder)
    if act == 'unknown': skipped.append((folder,'unknown')); continue
    wf = [f for f in os.listdir(fpath) if f.lower()=='wearable.csv']
    sf = [f for f in os.listdir(fpath) if f.lower()=='sc.csv']
    if not wf: skipped.append((folder,'no wearable')); continue
    try:
        imu = read_wearable(os.path.join(fpath, wf[0]))
        sc  = read_sc(os.path.join(fpath, sf[0])) if sf else pd.DataFrame(columns=['timestamp','spo2','hr'])
        if len(imu) < 10: skipped.append((folder,f'{len(imu)} rows')); continue
        segments.append({'activity':act,'participant':pid,'imu':imu,'sc':sc})
    except Exception as e:
        skipped.append((folder, str(e)))

print(f"Segmentos carregados : {len(segments)}  |  Ignorados : {len(skipped)}")
act_counts = pd.Series([s['activity'] for s in segments]).value_counts()
print("Atividades:\n" + act_counts.to_string() + "\n")

# =============================================================================
# 2. FEATURES
# =============================================================================
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report

try:
    from xgboost import XGBClassifier; USE_XGB = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier; USE_XGB = False
    print("XGBoost não disponível — a usar RandomForest.")

def extract_features(imu_df, sc_df):
    feats = {}
    for col in imu_df.fillna(imu_df.median(numeric_only=True)).columns:
        x = imu_df[col].fillna(0).to_numpy(dtype=float)
        feats[f'{col}_mean']   = np.mean(x)
        feats[f'{col}_std']    = np.std(x)
        feats[f'{col}_min']    = np.min(x)
        feats[f'{col}_max']    = np.max(x)
        feats[f'{col}_p25']    = np.percentile(x, 25)
        feats[f'{col}_p75']    = np.percentile(x, 75)
        feats[f'{col}_iqr']    = np.percentile(x,75) - np.percentile(x,25)
        feats[f'{col}_energy'] = np.mean(x**2)
    for sig in ['hr','spo2']:
        if sc_df is not None and sig in sc_df.columns and len(sc_df) > 0:
            x = sc_df[sig].dropna().to_numpy(dtype=float)
            if len(x) > 0:
                feats[f'sc_{sig}_mean'] = np.mean(x); feats[f'sc_{sig}_std']  = np.std(x)
                feats[f'sc_{sig}_min']  = np.min(x);  feats[f'sc_{sig}_max']  = np.max(x)
                continue
        feats[f'sc_{sig}_mean'] = feats[f'sc_{sig}_std'] = 0
        feats[f'sc_{sig}_min']  = feats[f'sc_{sig}_max'] = 0
    feats['n_samples'] = len(imu_df)
    return feats

imu_cols_train = ['ax','ay','az','gx','gy','gz']

def routine_to_windows(rt):
    rows, labels, hours = [], [], []
    n = len(rt)
    for start in range(0, n - WINDOW_SIZE + 1, STEP_SIZE):
        w   = rt.iloc[start:start + WINDOW_SIZE]
        sc_ = pd.DataFrame({'timestamp': range(WINDOW_SIZE),
                            'hr':        w['hr'].values,
                            'spo2':      w['spo2'].values})
        rows.append(extract_features(w[imu_cols_train], sc_))
        labels.append(w['activity_true'].mode()[0])
        hours.append(float(w['t_s'].mean() / 3600.))
    return rows, labels, hours

# =============================================================================
# 3. TEMPLATE DA ROTINA
# =============================================================================
NORMAL_ROUTINE = [
    (0.0,  'Laying',              3600, 7200),
    (7.0,  'Standing',              30,   90),
    (7.1,  'Washing Teeth',         60,  150),
    (7.3,  'Showering',            300,  600),
    (7.8,  'Walking',              120,  300),
    (8.0,  'Seating',              600, 1200),
    (8.5,  'Eating with hands',    300,  600),
    (9.0,  'Walking',               60,  180),
    (9.5,  'Seating',             3600, 5400),
    (12.0, 'Walking',              120,  300),
    (12.5, 'Eating with cutlery',  600, 1200),
    (13.0, 'Walking',              120,  300),
    (13.5, 'Seating',             3600, 5400),
    (17.0, 'Walking',              300,  600),
    (17.5, 'Light Movement',       300,  600),
    (18.0, 'Seating',              600, 1200),
    (19.0, 'Eating with cutlery',  600, 1200),
    (20.0, 'Seating',             1200, 2400),
    (21.5, 'Washing Hands',         20,   60),
    (21.6, 'Washing Teeth',         60,  150),
    (22.0, 'Laying',              3600, 7200),
]

# =============================================================================
# 4. POOLS DE SEGMENTOS (participant split — R2.4) E SIMULAÇÃO
# =============================================================================
print("─── 4. Pools de segmentos e simulação ───────────────────────────────")

def pool_from(segs):
    d = {}
    for s in segs:
        d.setdefault(s['activity'], []).append(s)
    return d

participants = sorted({s['participant'] for s in segments})
all_acts_data = sorted({s['activity'] for s in segments})

if PARTICIPANT_SPLIT and len(participants) > N_TEST_PARTICIPANTS:
    if TEST_PARTICIPANTS:
        test_p = set(TEST_PARTICIPANTS)
    else:
        # Escolhe o conjunto de teste que (1) deixa todas as atividades no treino
        # e (2) cobre o maior nº de atividades no teste.
        _rng, best = random.Random(RANDOM_SEED), None
        for _ in range(2000):
            cand  = frozenset(_rng.sample(participants, N_TEST_PARTICIPANTS))
            te_a  = {s['activity'] for s in segments if s['participant'] in cand}
            tr_a  = {s['activity'] for s in segments if s['participant'] not in cand}
            score = (len(tr_a) == len(all_acts_data), len(te_a))
            if best is None or score > best[0]:
                best = (score, cand)
        test_p = set(best[1])
    train_segs = [s for s in segments if s['participant'] not in test_p]
    test_segs  = [s for s in segments if s['participant'] in test_p]
else:
    test_p = set()
    train_segs = test_segs = segments

POOL_TRAIN = pool_from(train_segs)
POOL_TEST  = pool_from(test_segs)

print(f"Participantes: {len(participants)}  | teste: {sorted(test_p) if test_p else '— (sem split)'}")
print(f"Segmentos treino: {len(train_segs)}  | teste: {len(test_segs)}")
print(f"  {'Atividade':<24} {'treino':>6} {'teste':>6}")
for a in all_acts_data:
    print(f"  {a:<24} {len(POOL_TRAIN.get(a,[])):>6} {len(POOL_TEST.get(a,[])):>6}")
_missing_test = [a for a in all_acts_data if a not in POOL_TEST]
if _missing_test:
    print(f"  ATENÇÃO: sem segmentos de teste para {_missing_test} → usa FALLBACK")
print()

FALLBACK = {
    'Eating with hands':   ['Eating with cutlery','Light Movement','Seating'],
    'Eating with cutlery': ['Eating with hands','Light Movement','Seating'],
    'Showering':           ['Washing Teeth','Washing Hands','Light Movement'],
    'Washing Teeth':       ['Washing Hands','Showering','Light Movement'],
    'Washing Hands':       ['Washing Teeth','Showering','Light Movement'],
    'Light Movement':      ['Walking','Standing','Seating'],
}

def get_seg(act, pool):
    if act in pool: return random.choice(pool[act])
    for a in FALLBACK.get(act,[]):
        if a in pool: return random.choice(pool[a])
    return random.choice([s for v in pool.values() for s in v])

HR_RANGES = {
    'Laying':(55,65),'Seating':(60,75),'Standing':(65,80),
    'Walking':(80,100),'Light Movement':(75,95),
    'Eating with hands':(65,80),'Eating with cutlery':(65,80),
    'Washing Hands':(65,75),'Washing Teeth':(65,75),'Showering':(70,85)
}

def sample_imu_sc(seg, n):
    imu = seg['imu'].to_numpy(dtype=float)
    if len(imu) == 0:   imu = np.zeros((n,6))
    elif len(imu) < n:  imu = np.tile(imu, (int(np.ceil(n/len(imu))),1))[:n]
    else:               s=random.randint(0,len(imu)-n); imu=imu[s:s+n]
    sc = seg['sc']
    if sc is not None and 'hr' in sc.columns and len(sc)>0:
        hr=sc['hr'].dropna().to_numpy(dtype=float)
        sp=sc['spo2'].dropna().to_numpy(dtype=float)
        if len(hr)<n: hr=np.resize(hr,n); sp=np.resize(sp,n)
        else: s=random.randint(0,len(hr)-n); hr=hr[s:s+n]; sp=sp[s:s+n]
    else:
        lo,hi = HR_RANGES.get(seg['activity'],(65,85))
        hr = np.random.uniform(lo,hi,n)
        sp = np.random.uniform(96,99,n)
    return imu, hr, sp

def build_routine(template, label='normal', anomaly_config=None, base_durs=None, pool=None):
    pool = POOL_TRAIN if pool is None else pool
    injs = sorted(anomaly_config or [], key=lambda x: x['step'])
    ii   = 0
    step_durs, step_aact, step_is_a, step_atype = [], [], [], []
    anomalous_steps = set()

    for si, (_, act, dmin, dmax) in enumerate(template):
        aact, is_a, atype = act, False, None
        if ii < len(injs) and injs[ii]['step'] == si:
            inj = injs[ii]; ii += 1
            if inj['type'] == 'duration':
                dur = inj['duration_s']; is_a = True; atype = 'duration'
                anomalous_steps.add(si)
            elif inj['type'] == 'wrong_activity':
                aact = inj['replace_with']; is_a = True; atype = 'wrong_activity'
                dur  = base_durs[si] if base_durs is not None else random.randint(dmin, dmax)
            else:
                dur = base_durs[si] if base_durs is not None else random.randint(dmin, dmax)
        else:
            dur = base_durs[si] if base_durs is not None else random.randint(dmin, dmax)
        step_durs.append(dur); step_aact.append(aact)
        step_is_a.append(is_a); step_atype.append(atype)

    if base_durs is not None:
        target = sum(base_durs); delta = target - sum(step_durs)
        if delta != 0:
            absorber = max(
                (i for i in range(len(template)) if i not in anomalous_steps),
                key=lambda i: step_durs[i])
            step_durs[absorber] = max(1, step_durs[absorber] + delta)

    records, t = [], 0
    for si, (_, act, dmin, dmax) in enumerate(template):
        dur = step_durs[si]; aact = step_aact[si]
        is_a = step_is_a[si]; atype = step_atype[si]
        seg = get_seg(aact, pool); imu, hr, sp = sample_imu_sc(seg, dur)
        pad = np.zeros((dur, 6)); c = min(imu.shape[1], 6); pad[:, :c] = imu[:, :c]
        for i in range(dur):
            records.append({'t_s': t+i, 'hour': (t+i)/3600.,
                            'activity_true': act, 'activity_actual': aact,
                            'is_anomaly': is_a, 'anomaly_type': atype or 'none',
                            'ax': pad[i,0], 'ay': pad[i,1], 'az': pad[i,2],
                            'gx': pad[i,3], 'gy': pad[i,4], 'gz': pad[i,5],
                            'hr': hr[i], 'spo2': sp[i]})
        t += dur
    return pd.DataFrame(records).assign(routine_label=label), step_durs

# =============================================================================
# 5. TREINO DO CLASSIFICADOR (rotinas de participantes de treino)
# =============================================================================
print("─── 5. Treino do classificador ──────────────────────────────────────")
print(f"Gerando {N_TRAIN_ROUTINES} rotinas de treino...")
all_rows, all_labels, all_hours, all_groups = [], [], [], []
for ri in range(N_TRAIN_ROUTINES):
    rt, _ = build_routine(NORMAL_ROUTINE, 'normal', pool=POOL_TRAIN)
    r, l, h = routine_to_windows(rt)
    all_rows.extend(r); all_labels.extend(l)
    all_hours.extend(h); all_groups.extend([f'r{ri:03d}'] * len(r))

X = pd.DataFrame(all_rows).fillna(0)
y = pd.Series(all_labels)
g = pd.Series(all_groups)
print(f"Janelas de treino: {len(X):,}  (~{len(X)//N_TRAIN_ROUTINES} por rotina)")

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
tr, te = next(gss.split(X, y, groups=g))

scaler = StandardScaler()
le     = LabelEncoder()
X_tr   = scaler.fit_transform(X.iloc[tr])
X_te   = scaler.transform(X.iloc[te])
y_tr   = le.fit_transform(y.iloc[tr])
y_te   = le.transform(y.iloc[te])

clf = (XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                     subsample=0.9, colsample_bytree=0.9,
                     objective='multi:softprob', num_class=len(le.classes_),
                     eval_metric='mlogloss', tree_method='hist',
                     random_state=RANDOM_SEED)
       if USE_XGB else
       RandomForestClassifier(n_estimators=400, max_depth=8,
                              random_state=RANDOM_SEED, n_jobs=-1))

clf.fit(X_tr, y_tr)
y_pred_val = le.inverse_transform(clf.predict(X_te))
print("Validação (20% das rotinas de TREINO — mesmos participantes):")
print(classification_report(y.iloc[te], y_pred_val))
joblib.dump({'clf':clf,'scaler':scaler,'le':le,'cols':list(X.columns),
             'test_participants': sorted(test_p)},
            os.path.join(OUTPUT_DIR,'classifier.pkl'))
feature_cols = list(X.columns)

def classify_routine(routine_df):
    """Classificação por janela deslizante — features por janela, predict em batch."""
    imu_cols = ['ax','ay','az','gx','gy','gz']
    feats, meta = [], []
    n = len(routine_df)
    for start in range(0, n-WINDOW_SIZE+1, STEP_SIZE):
        w = routine_df.iloc[start:start+WINDOW_SIZE]
        feats.append(extract_features(w[imu_cols],
                     w[['hr','spo2']].assign(timestamp=range(WINDOW_SIZE))))
        at = w.loc[w['is_anomaly'],'anomaly_type'].unique()
        meta.append({
            't_start':w['t_s'].iloc[0], 'hour':w['t_s'].mean()/3600.,
            'activity_true':w['activity_true'].mode()[0],
            'activity_actual':w['activity_actual'].mode()[0],
            'is_anomaly':w['is_anomaly'].any(),
            'anomaly_type':at[0] if len(at) else 'none',
            'hr_mean':w['hr'].mean(), 'spo2_mean':w['spo2'].mean(),
        })
    fv = pd.DataFrame(feats).fillna(0)
    for c in feature_cols:
        if c not in fv.columns: fv[c] = 0
    preds = le.inverse_transform(clf.predict(scaler.transform(fv[feature_cols])))
    out = pd.DataFrame(meta)
    out['activity_predicted'] = preds
    return out

# =============================================================================
# 6. LSTM AUTOENCODER — DETETOR COMPORTAMENTAL
# =============================================================================
print("─── 6. LSTM Autoencoder — Detetor Comportamental ───────────────────")

def smooth_labels(preds, k):
    if k <= 1: return list(preds)
    half, out = k // 2, []
    for i in range(len(preds)):
        s = max(0, i - half); e = min(len(preds), i + half + 1)
        out.append(pd.Series(preds[s:e]).mode()[0])
    return out

def smooth_predictions(wdf, k=SMOOTH_K):
    return smooth_labels(wdf['activity_predicted'].tolist(), k)

train_preds_raw = le.inverse_transform(clf.predict(scaler.transform(X)))
train_smoothed_labels = []
_groups_arr = np.array(all_groups)
for ri in range(N_TRAIN_ROUTINES):
    preds_ri = train_preds_raw[_groups_arr == f'r{ri:03d}'].tolist()
    train_smoothed_labels.extend(smooth_labels(preds_ri, SMOOTH_K))

all_act_classes = sorted(set(train_smoothed_labels) | set(le.classes_))
act2idx = {a: i for i, a in enumerate(all_act_classes)}
idx2act = {i: a for a, i in act2idx.items()}
N_ACT_CLASSES = len(all_act_classes)
print(f"Classes de atividade para LSTM AE: {N_ACT_CLASSES}  → {all_act_classes}")

train_indices = np.array([act2idx[a] for a in train_smoothed_labels])

def extract_sequences(indices, groups, n_routines, seq_len):
    seqs = []
    for ri in range(n_routines):
        idx_ri = indices[groups == f'r{ri:03d}']
        for i in range(len(idx_ri) - seq_len + 1):
            seqs.append(idx_ri[i:i + seq_len])
    return np.array(seqs, dtype=np.int32)

train_seqs = extract_sequences(train_indices, _groups_arr, N_TRAIN_ROUTINES, LSTM_SEQ_LEN)
print(f"Sequências de treino: {len(train_seqs):,}")

def make_context_seqs(idx):
    """Uma sequência de LSTM_SEQ_LEN por janela, terminando nessa janela.
    As primeiras LSTM_SEQ_LEN-1 janelas são preenchidas à esquerda:
      PAD_MODE='edge' → com a primeira atividade observada
      PAD_MODE='zero' → com o índice 0 (comportamento v10)"""
    n = len(idx)
    seqs = np.zeros((n, LSTM_SEQ_LEN), dtype=np.int32)
    fill = int(idx[0]) if (PAD_MODE == 'edge' and n) else 0
    for i in range(n):
        if i < LSTM_SEQ_LEN - 1:
            p = LSTM_SEQ_LEN - i - 1
            seqs[i, :p] = fill
            seqs[i, p:] = idx[:i+1]
        else:
            seqs[i] = idx[i - LSTM_SEQ_LEN + 1: i + 1]
    return seqs

def build_lstm_ae(n_classes, seq_len, embed_dim=8, lstm_units=64):
    inp = Input(shape=(seq_len,), name='activity_seq')
    x   = Embedding(n_classes, embed_dim, name='embedding')(inp)
    enc = LSTM(lstm_units, return_sequences=False, name='encoder')(x)
    x   = RepeatVector(seq_len, name='repeat')(enc)
    x   = LSTM(lstm_units, return_sequences=True, name='decoder')(x)
    out = TimeDistributed(Dense(n_classes, activation='softmax'), name='output')(x)
    model = Model(inp, out, name='LSTM_AE_behavioral')
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model

lstm_ae = build_lstm_ae(N_ACT_CLASSES, LSTM_SEQ_LEN, LSTM_EMBED_DIM, LSTM_UNITS)

# Split em ordem de rotina: as últimas ~20% das sequências pertencem às últimas
# ~30 rotinas de treino (sem sobreposição com as sequências de treino).
val_split = int(0.8 * len(train_seqs))
X_ae_tr, X_ae_val = train_seqs[:val_split], train_seqs[val_split:]
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
history = lstm_ae.fit(X_ae_tr, X_ae_tr[..., np.newaxis],
                      validation_data=(X_ae_val, X_ae_val[..., np.newaxis]),
                      epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH,
                      callbacks=[es], verbose=2)
print(f"Treino LSTM AE: {len(history.history['loss'])} épocas  "
      f"| val_loss final: {history.history['val_loss'][-1]:.4f}")

def last_pos_errors(seqs):
    preds = lstm_ae.predict(seqs, batch_size=512, verbose=0)
    last  = seqs[:, -1]
    prob  = preds[np.arange(len(seqs)), -1, last]
    return -np.log(prob + 1e-8), last

# ── Calibração — rotinas held-out (participantes de treino) ──────────────────
print(f"Classificando {N_CALIB} rotinas de calibração held-out...")
calib_windows = [classify_routine(build_routine(NORMAL_ROUTINE, 'normal', pool=POOL_TRAIN)[0])
                 for _ in range(N_CALIB)]

_err_cache = {}
def calib_errors(source, k):
    """Erros de reconstrução usados para calibrar thresholds.
      source='heldout' → N_CALIB rotinas normais não usadas no treino (v10)
      source='train'   → sequências de treino (calibração antiga, P95)"""
    key = (source, k)
    if key in _err_cache: return _err_cache[key]
    if source == 'train':
        errs, last = last_pos_errors(train_seqs)
    else:
        e_all, l_all = [], []
        for wc in calib_windows:
            idx = np.array([act2idx.get(a, 0) for a in smooth_predictions(wc, k)], dtype=np.int32)
            e, l = last_pos_errors(make_context_seqs(idx))
            e_all.append(e); l_all.append(l)
        errs, last = np.concatenate(e_all), np.concatenate(l_all)
    by_act = {a: errs[last == i] for a, i in act2idx.items()}
    _err_cache[key] = (errs, by_act)
    return _err_cache[key]

def get_thresholds(cfg):
    errs, by_act = calib_errors(cfg['calib'], cfg['k'])
    g = float(np.percentile(errs, cfg['p']))
    th = {a: (float(np.percentile(e, cfg['p'])) if cfg['per_class'] and len(e) >= 20 else g)
          for a, e in by_act.items()}
    return g, th

# =============================================================================
# 7. LIMITES DE DURAÇÃO (por ocorrência, ordem do template)
# =============================================================================
step_occur_bounds = defaultdict(list)
for _t, _act, _dmin, _dmax in NORMAL_ROUTINE:
    step_occur_bounds[_act].append((_dmin * DUR_MARGIN_LO, _dmax * DUR_MARGIN_HI))

# =============================================================================
# 8. DETEÇÃO DE ANOMALIAS (configurável para ablação)
# =============================================================================
DEFAULT_CFG = dict(k=SMOOTH_K, use_beh=True, use_dur=True, cascade=CASCADE_SUPPRESS,
                   p=LSTM_AE_THRESH_P, per_class=True, calib='heldout', oracle=False)

def _blocks(labels):
    """[(start, end, label)] de blocos contíguos."""
    out, cur, bs = [], None, 0
    for i, a in enumerate(labels):
        if a != cur:
            if cur is not None: out.append((bs, i, cur))
            cur, bs = a, i
    if cur is not None: out.append((bs, len(labels), cur))
    return out

def detect_anomalies(wdf, cfg=None):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    r = wdf.copy().reset_index(drop=True)
    n = len(r)
    if cfg['oracle']:   # HAR perfeito: isola o desempenho dos detetores
        r['activity_predicted'] = r['activity_actual']
    r['activity_smoothed'] = smooth_predictions(r, cfg['k'])
    det_dur = np.zeros(n, bool); det_beh = np.zeros(n, bool); score = np.zeros(n)

    # ── Detetor de duração ──
    blocks = _blocks(r[DUR_SIGNAL].tolist())
    if cfg['use_dur']:
        occ = defaultdict(int)
        t0 = r['t_start'].values
        for s, e, act in blocks:
            dur = t0[e-1] - t0[s] + WINDOW_SIZE
            bl  = step_occur_bounds.get(act, [(0, np.inf)])
            lo, hi = bl[min(occ[act], len(bl) - 1)]
            occ[act] += 1
            if dur < lo or dur > hi:
                det_dur[s:e] = True
                sc = max(0., 1. - dur / max(lo, 1.)) if dur < lo else min(1., (dur - hi) / max(hi, 1.))
                score[s:e] = np.maximum(score[s:e], sc)

    # ── Supressão em cascata (por bloco) ──
    suppressed = np.zeros(n, bool)
    if cfg['cascade'] and cfg['use_dur']:
        for s, e, _ in blocks:
            if det_dur[s:e].any(): suppressed[s:e] = True

    # ── Detetor comportamental (LSTM AE) ──
    if cfg['use_beh']:
        g_thr, thr = get_thresholds(cfg)
        idx  = np.array([act2idx.get(a, 0) for a in r[BEHAVIORAL_SIGNAL]], dtype=np.int32)
        act_ = np.where(~suppressed)[0]
        if len(act_):
            err, last = last_pos_errors(make_context_seqs(idx)[act_])
            for k_, i in enumerate(act_):
                t = thr.get(idx2act.get(int(last[k_]), ''), g_thr)
                if err[k_] > t:
                    det_beh[i] = True
                    score[i] = max(score[i], min(1.0, (err[k_] - t) / max(t, 1e-9)))

    r['detected_duration']   = det_dur
    r['detected_behavioral'] = det_beh
    r['detected_any']        = det_dur | det_beh
    r['anomaly_score']       = score
    return r

def pair_metrics(dn_i, da_i):
    y_t = da_i['is_anomaly'].astype(int).values
    y_p = da_i['detected_any'].astype(int).values
    tp_ = int(((y_t==1)&(y_p==1)).sum()); fp_ = int(((y_t==0)&(y_p==1)).sum())
    fn_ = int(((y_t==1)&(y_p==0)).sum()); fa_ = int(dn_i['detected_any'].sum())
    prec_ = tp_/(tp_+fp_) if tp_+fp_ else 0.0
    rec_  = tp_/(tp_+fn_) if tp_+fn_ else 0.0
    f1_   = 2*prec_*rec_/(prec_+rec_) if prec_+rec_ else 0.0
    sub_wa = da_i[da_i['anomaly_type']=='wrong_activity']
    sub_du = da_i[da_i['anomaly_type']=='duration']
    return dict(tp=tp_, fp=fp_, fn=fn_, fa=fa_, precision=prec_, recall=rec_, f1=f1_,
                far=fa_/len(dn_i) if len(dn_i) else 0.0,
                dr_wrong_activity=sub_wa['detected_any'].mean() if len(sub_wa) else np.nan,
                dr_duration=sub_du['detected_any'].mean() if len(sub_du) else np.nan)

# =============================================================================
# 9. GERAÇÃO DE ANOMALIAS (igual à v10)
# =============================================================================
_unique_acts = sorted(set(act for _, act, _, _ in NORMAL_ROUTINE))
_DISTINCTIVE_ACTS = {'Eating with cutlery', 'Eating with hands',
                     'Showering', 'Light Movement', 'Walking'}
_wa_candidate_steps = [i for i, (_, act, dmin, _) in enumerate(NORMAL_ROUTINE)
                       if dmin >= 300 and act in _DISTINCTIVE_ACTS]
_dur_long_steps  = list(range(len(NORMAL_ROUTINE)))
_dur_short_steps = [i for i, (_, _, dmin, _) in enumerate(NORMAL_ROUTINE) if dmin >= 1200]

def random_anomaly_config(n_wrong=2):
    wa_steps = random.sample(_wa_candidate_steps, min(n_wrong, len(_wa_candidate_steps)))
    used, config = set(wa_steps), []
    for s in wa_steps:
        orig = NORMAL_ROUTINE[s][1]
        config.append({'step': s, 'type': 'wrong_activity',
                       'replace_with': random.choice([a for a in _unique_acts if a != orig])})
    long_avail = [s for s in _dur_long_steps if s not in used]
    if long_avail:
        s = random.choice(long_avail)
        config.append({'step': s, 'type': 'duration', 'duration_s': int(NORMAL_ROUTINE[s][3] * 4.5)})
        used.add(s)
    short_avail = [s for s in _dur_short_steps if s not in used]
    if short_avail:
        s = random.choice(short_avail)
        config.append({'step': s, 'type': 'duration',
                       'duration_s': max(1, int(NORMAL_ROUTINE[s][2] * 0.05))})
    return sorted(config, key=lambda x: x['step'])

# =============================================================================
# 10. AVALIAÇÃO MULTI-ROTINA (participantes de teste)
# =============================================================================
print(f"\n─── 10. Avaliação multi-rotina (N={N_EVAL_ROUTINES}, participantes de teste) ───")
eval_pairs = []
for ei in range(N_EVAL_ROUTINES):
    rn_ei, base_ei = build_routine(NORMAL_ROUTINE, 'normal', pool=POOL_TEST)
    ac_ei          = random_anomaly_config()
    ra_ei, _       = build_routine(NORMAL_ROUTINE, 'anomalous', ac_ei,
                                   base_durs=base_ei, pool=POOL_TEST)
    eval_pairs.append((classify_routine(rn_ei), classify_routine(ra_ei), ac_ei))
    print(f"  Par {ei+1:02d}/{N_EVAL_ROUTINES} classificado")

eval_acc = [(wn_['activity_predicted'] == wn_['activity_actual']).mean() for wn_, _, _ in eval_pairs]
print(f"\nPrecisão XGBoost (janelas, rotinas normais de TESTE): "
      f"{np.mean(eval_acc):.1%} ± {np.std(eval_acc):.1%}")

def run_cfg(cfg=None):
    rows = []
    for wn_, wa_, _ in eval_pairs:
        rows.append(pair_metrics(detect_anomalies(wn_, cfg), detect_anomalies(wa_, cfg)))
    return pd.DataFrame(rows)

# Matriz de confusão do XGBoost nas rotinas normais de teste (linhas = real)
_cm_true = pd.concat([w['activity_actual'] for w, _, _ in eval_pairs], ignore_index=True)
_cm_pred = pd.concat([w['activity_predicted'] for w, _, _ in eval_pairs], ignore_index=True)
cm = pd.crosstab(_cm_true, _cm_pred, normalize='index').round(3)
cm.to_csv(os.path.join(OUTPUT_DIR, 'confusion_matrix_test.csv'))
print("\nMatriz de confusão (teste, proporção por linha = atividade real):")
with pd.option_context('display.width', 250, 'display.max_columns', 20):
    print(cm.to_string())
print("Recall por atividade (teste):")
for a in cm.index:
    print(f"  {a:<24} {cm.loc[a, a] if a in cm.columns else 0:.1%}")

mdf = run_cfg()
mdf.to_csv(os.path.join(OUTPUT_DIR, 'multi_eval_metrics.csv'), index=False)

def summ(df, col):
    v = df[col].dropna()
    return f"{v.mean():.3f} ± {v.std():.3f}" if len(v) else "n/a"

print(f"\n  RESUMO (configuração completa)")
for col in ['precision','recall','f1','far','dr_wrong_activity','dr_duration']:
    print(f"  {col:<20} {summ(mdf, col)}")

# =============================================================================
# 10b. ABLAÇÃO (R2.6) — mesmos pares de avaliação para todas as variantes
# =============================================================================
if RUN_ABLATION:
    print("\n─── 10b. Ablação ────────────────────────────────────────────────")
    ABLATIONS = [
        ('Full system (P97 held-out, per-class, k=5, cascade)', {}),
        ('Behavioural detector only (LSTM AE)',   dict(use_dur=False)),
        ('Duration detector only',                dict(use_beh=False)),
        ('Without cascade suppression',           dict(cascade=False)),
        ('Smoothing k=1 (none)',                  dict(k=1)),
        ('Smoothing k=3',                         dict(k=3)),
        ('Smoothing k=7',                         dict(k=7)),
        ('Threshold P90',                         dict(p=90)),
        ('Threshold P95',                         dict(p=95)),
        ('Threshold P99',                         dict(p=99)),
        ('Global threshold (not per-class)',      dict(per_class=False)),
        ('Training-data calibration (P95)',       dict(calib='train', p=95)),
        ('Oracle activity labels (perfect HAR)',  dict(oracle=True)),
    ]
    abl_rows = []
    for name, cfg in ABLATIONS:
        d = run_cfg(cfg)
        row = {'variant': name}
        for col in ['precision','recall','f1','far','dr_wrong_activity','dr_duration']:
            row[f'{col}_mean'] = d[col].mean(); row[f'{col}_std'] = d[col].std()
        abl_rows.append(row)
        print(f"  {name:<52} F1={row['f1_mean']:.3f}±{row['f1_std']:.3f}  "
              f"Rec={row['recall_mean']:.3f}  FAR={row['far_mean']:.1%}")
    abl_df = pd.DataFrame(abl_rows)
    abl_df.to_csv(os.path.join(OUTPUT_DIR, 'ablation_results.csv'), index=False)
    print("  → ablation_results.csv")

# =============================================================================
# 10c. SEVERIDADE DAS ANOMALIAS DE DURAÇÃO (R2.5)
# =============================================================================
if RUN_SEVERITY:
    print("\n─── 10c. Severidade das anomalias de duração ───────────────────")
    sev_rows = []
    for kind, factors, pool_steps, ref_col in [
            ('extension', SEV_LONG_FACTORS,  _dur_long_steps,  3),
            ('truncation', SEV_SHORT_FACTORS, _dur_short_steps, 2)]:
        for f in factors:
            for _ in range(N_SEVERITY):
                rn_s, base_s = build_routine(NORMAL_ROUTINE, 'normal', pool=POOL_TEST)
                s = random.choice(pool_steps)
                dur_s = max(1, int(NORMAL_ROUTINE[s][ref_col] * f))
                ra_s, _ = build_routine(NORMAL_ROUTINE, 'anomalous',
                                        [{'step': s, 'type': 'duration', 'duration_s': dur_s}],
                                        base_durs=base_s, pool=POOL_TEST)
                da_s = detect_anomalies(classify_routine(ra_s))
                sub  = da_s[da_s['is_anomaly']]
                sev_rows.append(dict(
                    kind=kind, factor=f, step=s, activity=NORMAL_ROUTINE[s][1], duration_s=dur_s,
                    within_bounds=(f <= DUR_MARGIN_HI) if kind == 'extension' else (f >= DUR_MARGIN_LO),
                    n_windows=len(sub),
                    event_detected_duration=bool(sub['detected_duration'].any()) if len(sub) else False,
                    event_detected_any=bool(sub['detected_any'].any()) if len(sub) else False,
                    window_dr_duration=sub['detected_duration'].mean() if len(sub) else np.nan,
                    window_dr_any=sub['detected_any'].mean() if len(sub) else np.nan))
            fr = pd.DataFrame([x for x in sev_rows if x['kind']==kind and x['factor']==f])
            print(f"  {kind:<10} ×{f:<5} event DR (duration)={fr['event_detected_duration'].mean():.0%}  "
                  f"(any)={fr['event_detected_any'].mean():.0%}")
    sev_df = pd.DataFrame(sev_rows)
    sev_df.to_csv(os.path.join(OUTPUT_DIR, 'severity_results.csv'), index=False)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, kind, lim, xl in [(axs[0], 'extension', DUR_MARGIN_HI, 'Injected duration / d_max'),
                              (axs[1], 'truncation', DUR_MARGIN_LO, 'Injected duration / d_min')]:
        agg = sev_df[sev_df['kind']==kind].groupby('factor')[
            ['event_detected_duration','event_detected_any']].mean().sort_index()
        ax.plot(agg.index, agg['event_detected_duration'], 'o-', label='Duration detector')
        ax.plot(agg.index, agg['event_detected_any'], 's--', label='Full system')
        ax.axvline(lim, color='grey', ls=':', label=f'Detection bound (×{lim})')
        ax.set_xlabel(xl); ax.set_title(kind.capitalize()); ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
    axs[0].set_ylabel('Event detection rate'); axs[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'severity_curve.png'), dpi=300)
    plt.close()
    print("  → severity_results.csv, severity_curve.png")

# =============================================================================
# 10d. ROTINAS NORMAIS PERTURBADAS (R2.3)
# =============================================================================
if RUN_PERTURB:
    print("\n─── 10d. Rotinas normais perturbadas ───────────────────────────")
    _middle = list(range(1, len(NORMAL_ROUTINE) - 1))   # nunca mexe no 1º/último Laying

    def perturbed_template(kind):
        t = list(NORMAL_ROUTINE)
        if kind == 'skip':
            del t[random.choice(_middle)]
        elif kind == 'swap':
            i = random.choice(_middle[:-1]); t[i], t[i+1] = t[i+1], t[i]
        elif kind == 'interrupt':
            cands = [i for i, (_, a, dmin, _) in enumerate(t) if a == 'Seating' and dmin >= 3600]
            i = random.choice(cands); h, a, dmin, dmax = t[i]
            t[i:i+1] = [(h, a, dmin // 2, dmax // 2), (h, 'Walking', 60, 180),
                        (h, a, dmin // 2, dmax // 2)]
        elif kind == 'jitter':
            t = [(h, a, max(1, int(dmin * 0.8)), int(dmax * 1.2)) for h, a, dmin, dmax in t]
        return t

    base_far = mdf['far']
    pert_rows = [dict(perturbation='none (template)', far_mean=base_far.mean(), far_std=base_far.std())]
    pert_detail = []
    for kind in ['skip', 'swap', 'interrupt', 'jitter']:
        fars, fars_b, fars_d = [], [], []
        for _ in range(N_PERTURB):
            rt, _ = build_routine(perturbed_template(kind), 'normal', pool=POOL_TEST)
            d = detect_anomalies(classify_routine(rt))
            fars.append(d['detected_any'].mean())
            fars_b.append(d['detected_behavioral'].mean()); fars_d.append(d['detected_duration'].mean())
            pert_detail.append(dict(perturbation=kind, far=fars[-1],
                                    far_behavioural=fars_b[-1], far_duration=fars_d[-1]))
        pert_rows.append(dict(perturbation=kind, far_mean=np.mean(fars), far_std=np.std(fars),
                              far_behavioural_mean=np.mean(fars_b), far_duration_mean=np.mean(fars_d)))
        print(f"  {kind:<10} FAR={np.mean(fars):.1%} ± {np.std(fars):.1%}  "
              f"(behavioural {np.mean(fars_b):.1%}, duration {np.mean(fars_d):.1%})")
    pd.DataFrame(pert_rows).to_csv(os.path.join(OUTPUT_DIR, 'perturbation_results.csv'), index=False)
    pd.DataFrame(pert_detail).to_csv(os.path.join(OUTPUT_DIR, 'perturbation_detail.csv'), index=False)
    print("  → perturbation_results.csv")

# =============================================================================
# 11. CSVs E GRÁFICO DE UM PAR REPRESENTATIVO
# =============================================================================
wn0, wa0, _ = eval_pairs[0]
dn, da = detect_anomalies(wn0), detect_anomalies(wa0)
dn.to_csv(os.path.join(OUTPUT_DIR,'detecao_normal.csv'),    index=False)
da.to_csv(os.path.join(OUTPUT_DIR,'detecao_anomalous.csv'), index=False)

all_acts_plot = sorted(set(dn['activity_predicted']) | set(da['activity_predicted']))
_tab20    = plt.get_cmap('tab20')
act_color = {a: _tab20(i % 20) for i,a in enumerate(all_acts_plot)}
fig, axes = plt.subplots(3,1,figsize=(18,14))
def draw_tl(ax, df, title, mark=False):
    for _, r in df.iterrows():
        ax.bar(r['hour'],1,width=STEP_SIZE/3600,
               color=act_color.get(r['activity_predicted'],'grey'), align='edge',alpha=0.8)
        if mark and r['detected_any']:
            ax.bar(r['hour'],1,width=STEP_SIZE/3600,color='none',
                   edgecolor='red',linewidth=1.5,align='edge')
    ax.set_title(title,fontsize=11); ax.set_yticks([]); ax.set_xlabel('Hora')
    p = [mpatches.Patch(color=act_color[a],label=a) for a in all_acts_plot]
    if mark: p.append(mpatches.Patch(edgecolor='red',facecolor='none',lw=1.5,label='Detetada'))
    ax.legend(handles=p,loc='upper right',fontsize=7,ncol=2)
draw_tl(axes[0], dn, 'Rotina Normal — Atividade Classificada')
draw_tl(axes[1], da, 'Rotina Anómala — Atividade Classificada + Deteções', True)
ax = axes[2]; ax.set_title('Score de Anomalia — Rotina Anómala',fontsize=11)
ax.fill_between(da['hour'],da['anomaly_score'],alpha=0.4,color='red')
for _, grp in da[da['is_anomaly']].groupby((da['is_anomaly'] != da['is_anomaly'].shift()).cumsum()):
    ax.axvspan(grp['hour'].iloc[0], grp['hour'].iloc[-1], alpha=0.15, color='orange')
ax.set_ylim(0,1.05); ax.set_xlabel('Hora')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'resultados_pipeline.png'),dpi=150,bbox_inches='tight')
plt.close()

print(f"\n{'='*60}")
print(f"  PIPELINE CONCLUÍDO — v{VERSION}   resultados em {OUTPUT_DIR}")
print(f"{'='*60}\n")
