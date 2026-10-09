"""
Analysis of accuracy-test logs produced by 03 inference/infer.py.

Single log  -> accuracy, confusion matrix, per-class precision/recall/F1, macro-F1,
               segment-bootstrap 95% confidence intervals, recovery delay, calibration.
Several logs -> one report per log, plus a side-by-side summary and pairwise
               McNemar tests on the windows the models share.

Usage:
    python analysis.py <log.csv>                  # one model
    python analysis.py <log_a.csv> <log_b.csv>    # compare models
Log names may be given as bare file names; they are looked up in 03 inference/logs.
"""

import math
import os
import sys
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd

BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 0
CALIBRATION_BINS = 10
SIGNIFICANCE_LEVEL = 0.05
LINE_WIDTH = 78

_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(_UTILS_DIR, '..', '03 inference', 'logs')
REPORTS_DIR = os.path.join(_UTILS_DIR, '..', '03 inference', 'analysis_reports')


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def load_log(log_file_path):
    """
    Read an accuracy log into a DataFrame with one row per classified window.

    Columns: Timestamp (datetime), Label, Prediction, Confidence (0-1 or NaN),
    Correct (bool), Segment (int; a new segment starts whenever Label changes).
    Non-data lines appended to the log (e.g. a copied report) are skipped.
    Returns None if the file is missing or holds no usable rows.
    """
    if not os.path.exists(log_file_path):
        print(f"Error: Log file not found: {log_file_path}")
        return None

    raw = pd.read_csv(log_file_path, on_bad_lines='skip', dtype=str)
    cols = {c.strip().lower(): c for c in raw.columns}
    if 'label' not in cols or 'prediction' not in cols:
        print(f"Error: '{log_file_path}' needs 'Label' and 'Prediction' columns. "
              f"Found: {list(raw.columns)}")
        return None

    df = pd.DataFrame({
        'Timestamp': pd.to_datetime(raw[cols['timestamp']], errors='coerce')
        if 'timestamp' in cols else pd.NaT,
        'Label': raw[cols['label']].str.strip(),
        'Prediction': raw[cols['prediction']].str.strip(),
        'Confidence': pd.to_numeric(raw[cols['confidence']], errors='coerce')
        if 'confidence' in cols else np.nan,
    })
    df = df.dropna(subset=['Label', 'Prediction'])
    if 'timestamp' in cols:
        df = df.dropna(subset=['Timestamp'])
    df = df.reset_index(drop=True)
    if df.empty:
        print(f"Warning: No classified windows found in '{log_file_path}'.")
        return None

    if df['Confidence'].max() > 1.0:  # logged as a percentage
        df['Confidence'] = df['Confidence'] / 100.0
    df['Correct'] = df['Label'].str.lower() == df['Prediction'].str.lower()
    df['Segment'] = (df['Label'] != df['Label'].shift()).cumsum()
    return df


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def confusion_matrix(df, classes):
    """Counts with true classes as rows and predicted classes as columns."""
    return (pd.crosstab(df['Label'], df['Prediction'])
            .reindex(index=classes, columns=classes, fill_value=0))


def per_class_metrics(cm):
    """Precision, recall, F1 and support per class from a confusion matrix."""
    rows = []
    for c in cm.index:
        tp = cm.loc[c, c]
        predicted = cm[c].sum()
        actual = cm.loc[c].sum()
        precision = tp / predicted if predicted else np.nan
        recall = tp / actual if actual else np.nan
        if np.isnan(precision) or np.isnan(recall) or precision + recall == 0:
            f1 = 0.0 if actual else np.nan
        else:
            f1 = 2 * precision * recall / (precision + recall)
        rows.append({'Class': c, 'Precision': precision, 'Recall': recall,
                     'F1': f1, 'Support': int(actual)})
    return pd.DataFrame(rows)


def macro_f1(labels, predictions):
    """Unweighted mean F1 over the classes present in the ground truth."""
    scores = []
    for c in np.unique(labels):
        tp = np.sum((labels == c) & (predictions == c))
        fp = np.sum((labels != c) & (predictions == c))
        fn = np.sum((labels == c) & (predictions != c))
        scores.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return float(np.mean(scores))


def bootstrap_ci(df):
    """
    95% percentile intervals for accuracy and macro-F1, resampling whole
    segments with replacement (windows within a segment are correlated).
    """
    segments = [(g['Label'].to_numpy(), g['Prediction'].to_numpy())
                for _, g in df.groupby('Segment')]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    accs, f1s = [], []
    for _ in range(BOOTSTRAP_ITERATIONS):
        picks = rng.integers(0, len(segments), len(segments))
        y = np.concatenate([segments[i][0] for i in picks])
        p = np.concatenate([segments[i][1] for i in picks])
        accs.append(np.mean(y == p))
        f1s.append(macro_f1(y, p))
    return {
        'accuracy': tuple(np.percentile(accs, [2.5, 97.5])),
        'macro_f1': tuple(np.percentile(f1s, [2.5, 97.5])),
    }


def recovery_by_segment(df):
    """
    For each segment: windows, segment accuracy, and how it started.
    Start is 'correct' (first window right), 'recovered' (first window wrong,
    later right; delay = time from first window to first correct window), or
    'never' (no window correct).
    """
    rows = []
    for seg, g in df.groupby('Segment'):
        hits = np.flatnonzero(g['Correct'].to_numpy())
        if len(hits) == 0:
            start, delay = 'never', np.nan
        elif hits[0] == 0:
            start, delay = 'correct', 0.0
        else:
            start = 'recovered'
            t0, t1 = g['Timestamp'].iloc[0], g['Timestamp'].iloc[hits[0]]
            delay = (t1 - t0).total_seconds() if pd.notna(t0) and pd.notna(t1) else np.nan
        rows.append({'Segment': int(seg), 'Label': g['Label'].iloc[0], 'Windows': len(g),
                     'Accuracy': g['Correct'].mean(), 'Start': start, 'Delay': delay})
    return pd.DataFrame(rows)


def recovery_by_class(seg_df, classes):
    """Segment outcomes and mean late-recovery delay per class."""
    rows = []
    for c in classes:
        s = seg_df[seg_df['Label'] == c]
        late = s.loc[s['Start'] == 'recovered', 'Delay'].dropna()
        rows.append({
            'Class': c,
            'Segments': len(s),
            'Correct at start': int((s['Start'] == 'correct').sum()),
            'Recovered late': int((s['Start'] == 'recovered').sum()),
            'Never recovered': int((s['Start'] == 'never').sum()),
            'Mean delay': late.mean() if len(late) else np.nan,
        })
    return pd.DataFrame(rows)


def calibration(df):
    """Reliability table and Expected Calibration Error from the Confidence column."""
    d = df.dropna(subset=['Confidence'])
    if d.empty:
        return None, np.nan
    bins = np.minimum((d['Confidence'] * CALIBRATION_BINS).astype(int), CALIBRATION_BINS - 1)
    rows, ece = [], 0.0
    for b, g in d.groupby(bins):
        conf, acc = g['Confidence'].mean(), g['Correct'].mean()
        ece += len(g) / len(d) * abs(acc - conf)
        rows.append({'Bin': f'{b * 10}-{b * 10 + 10}%', 'Windows': len(g),
                     'Mean confidence': conf, 'Accuracy': acc, 'Gap': conf - acc})
    return pd.DataFrame(rows), ece


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from the two discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


# ----------------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------------

def _pct(x):
    return '-' if pd.isna(x) else f'{100 * x:.1f}%'


def _sec(x):
    return '-' if pd.isna(x) else f'{x:.1f} s'


def _table(headers, rows, left=1):
    """Plain-text table; the first `left` columns are left-aligned, the rest right-aligned."""
    cells = [[str(h) for h in headers]] + [[str(v) for v in r] for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]

    def fmt(r):
        return '  '.join(v.ljust(w) if i < left else v.rjust(w)
                         for i, (v, w) in enumerate(zip(r, widths)))

    return '\n'.join([fmt(cells[0]), '  '.join('-' * w for w in widths)]
                     + [fmt(r) for r in cells[1:]])


def _section(title, explanation):
    return f"\n{title}\n{'-' * len(title)}\n{explanation}\n"


def _model_name(log_path):
    name = os.path.splitext(os.path.basename(log_path))[0]
    if name.startswith('accuracy_test_'):
        name = name[len('accuracy_test_'):]
    parts = name.rsplit('_', 2)  # drop trailing _YYYY-MM-DD_HH-MM-SS
    return parts[0] if len(parts) == 3 else name


# ----------------------------------------------------------------------------
# Reports
# ----------------------------------------------------------------------------

def summarize(df):
    """All single-model metrics in one dict."""
    classes = sorted(set(df['Label']) | set(df['Prediction']))
    cm = confusion_matrix(df, classes)
    seg = recovery_by_segment(df)
    cal_table, ece = calibration(df)
    return {
        'classes': classes,
        'windows': len(df),
        'correct': int(df['Correct'].sum()),
        'accuracy': df['Correct'].mean(),
        'macro_f1': macro_f1(df['Label'].to_numpy(), df['Prediction'].to_numpy()),
        'ci': bootstrap_ci(df),
        'confusion': cm,
        'per_class': per_class_metrics(cm),
        'segments': seg,
        'recovery': recovery_by_class(seg, sorted(df['Label'].unique())),
        'calibration': cal_table,
        'ece': ece,
    }


def format_report(m):
    """Human-readable report for one model's summary dict."""
    acc_lo, acc_hi = m['ci']['accuracy']
    f1_lo, f1_hi = m['ci']['macro_f1']
    out = ['=' * LINE_WIDTH, 'ACCURACY TEST ANALYSIS', '=' * LINE_WIDTH]

    out.append(_section(
        '1. Summary',
        f'95% CI: bootstrap over whole segments ({BOOTSTRAP_ITERATIONS} resamples),\n'
        'since windows within a segment are correlated.'))
    out.append(_table(['Metric', 'Value', '95% CI'], [
        ['Windows classified', m['windows'], ''],
        ['Accuracy', f"{_pct(m['accuracy'])} ({m['correct']}/{m['windows']})",
         f'{_pct(acc_lo)} - {_pct(acc_hi)}'],
        ['Macro-F1', _pct(m['macro_f1']), f'{_pct(f1_lo)} - {_pct(f1_hi)}'],
    ]))

    out.append(_section(
        '2. Confusion matrix',
        'Rows = true class, columns = predicted class. Diagonal = correct windows.'))
    cm = m['confusion']
    out.append(_table(['True \\ Pred'] + list(cm.columns),
                      [[c] + cm.loc[c].tolist() for c in cm.index]))

    out.append(_section(
        '3. Per-class metrics',
        'Precision: of windows predicted as the class, share that were correct (false alarms).\n'
        'Recall: of windows of the class, share that were detected (missed detections).\n'
        'F1: harmonic mean of the two. Support: true windows of the class.'))
    pc = m['per_class']
    out.append(_table(['Class', 'Precision', 'Recall', 'F1', 'Support'],
                      [[r.Class, _pct(r.Precision), _pct(r.Recall), _pct(r.F1), r.Support]
                       for r in pc.itertuples()]))
    out.append(f"\nMacro-F1 (unweighted mean of class F1): {_pct(m['macro_f1'])}")

    out.append(_section(
        '4. Recovery after odor change',
        'Correct at start: first window of the segment right (delay 0).\n'
        'Recovered late: first window wrong, a later one right. Mean delay covers these only.\n'
        'Never recovered: no window of the segment classified correctly.'))
    rc = m['recovery']
    out.append(_table(['Class', 'Segments', 'Correct at start', 'Recovered late',
                       'Never recovered', 'Mean delay'],
                      [[r[0], r[1], r[2], r[3], r[4], _sec(r[5])]
                       for r in rc.itertuples(index=False)]))

    out.append('\nPer segment:')
    start_text = {'correct': 'correct', 'recovered': 'late', 'never': 'NEVER'}
    out.append(_table(['#', 'True class', 'Windows', 'Accuracy', 'Start', 'Delay'],
                      [[r.Segment, r.Label, r.Windows, _pct(r.Accuracy),
                        start_text[r.Start], _sec(r.Delay) if r.Start == 'recovered' else '-']
                       for r in m['segments'].itertuples()], left=2))

    out.append(_section(
        '5. Calibration',
        'Does confidence match accuracy? Gap = mean confidence - accuracy;\n'
        'positive = overconfident. ECE = window-weighted mean |gap| (0 is perfect).'))
    if m['calibration'] is None:
        out.append('No Confidence column in this log.')
    else:
        out.append(_table(['Confidence', 'Windows', 'Mean confidence', 'Accuracy', 'Gap'],
                          [[r.Bin, r.Windows, _pct(r[2]), _pct(r.Accuracy),
                            f'{100 * r.Gap:+.1f} pts']
                           for r in m['calibration'].itertuples(index=False)]))
        out.append(f"\nExpected Calibration Error (ECE): {_pct(m['ece'])}")

    out.append('\n' + '=' * LINE_WIDTH + '\n')
    return '\n'.join(out)


def analyze_accuracy_log(log_file_path):
    """
    Analyze one accuracy log, print the report, and return it as a string
    (None if the log cannot be read). Called by infer.py after each test.
    """
    df = load_log(log_file_path)
    if df is None:
        return None
    report = format_report(summarize(df))
    print(report)
    return report


def compare_logs(log_paths):
    """
    Report each log, then a side-by-side summary and pairwise McNemar tests.
    Windows are paired by start timestamp, so only windows present in both
    logs (with the same true label) are compared.
    Returns the full comparison text.
    """
    models = {}
    for path in log_paths:
        df = load_log(path)
        if df is not None:
            models[_model_name(path)] = (df, summarize(df))
    if len(models) < 2:
        print('Error: need at least two readable logs to compare.')
        return None

    out = ['=' * LINE_WIDTH, 'MODEL COMPARISON', '=' * LINE_WIDTH]
    out.append(_section(
        '1. Summary',
        'Overlapping 95% CIs mean the models cannot be ranked with confidence.'))
    rows = []
    for name, (_, m) in models.items():
        never = int(m['recovery']['Never recovered'].sum())
        rows.append([name, m['windows'],
                     f"{_pct(m['accuracy'])} [{_pct(m['ci']['accuracy'][0])}-{_pct(m['ci']['accuracy'][1])}]",
                     f"{_pct(m['macro_f1'])} [{_pct(m['ci']['macro_f1'][0])}-{_pct(m['ci']['macro_f1'][1])}]",
                     never, _pct(m['ece'])])
    out.append(_table(['Model', 'Windows', 'Accuracy [95% CI]', 'Macro-F1 [95% CI]',
                       'Never-recovered segs', 'ECE'], rows))

    out.append(_section(
        "2. McNemar's test (paired, exact)",
        'Only windows both models classified are used. "A only" = A right, B wrong.\n'
        f'p < {SIGNIFICANCE_LEVEL}: the accuracy difference is unlikely to be chance.\n'
        'Windows within a segment are correlated, so treat p-values as approximate.'))
    rows = []
    for (a, (da, _)), (b, (db, _)) in combinations(models.items(), 2):
        pair = da.merge(db, on=['Timestamp', 'Label'], suffixes=('_a', '_b'))
        only_a = int((pair['Correct_a'] & ~pair['Correct_b']).sum())
        only_b = int((~pair['Correct_a'] & pair['Correct_b']).sum())
        p = mcnemar_exact(only_a, only_b)
        verdict = 'significant' if p < SIGNIFICANCE_LEVEL else 'not significant'
        rows.append([f'{a} vs {b}', len(pair),
                     _pct(pair['Correct_a'].mean()), _pct(pair['Correct_b'].mean()),
                     only_a, only_b, '<0.001' if p < 0.001 else f'{p:.3f}', verdict])
    out.append(_table(['A vs B', 'Paired', 'Acc A', 'Acc B', 'A only', 'B only',
                       'p-value', 'Result'], rows))
    out.append('\n' + '=' * LINE_WIDTH + '\n')

    text = '\n'.join(out)
    for name, (_, m) in models.items():
        print(f'\nModel: {name}\n' + format_report(m))
    print(text)
    return text


def _resolve(path):
    if os.path.exists(path):
        return path
    return os.path.join(LOGS_DIR, path)


if __name__ == '__main__':
    paths = [_resolve(p) for p in sys.argv[1:]]
    if not paths:
        print(__doc__)
    elif len(paths) == 1:
        analyze_accuracy_log(paths[0])
    else:
        comparison = compare_logs(paths)
        if comparison:
            os.makedirs(REPORTS_DIR, exist_ok=True)
            stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            out_path = os.path.join(REPORTS_DIR, f'comparison_{stamp}.txt')
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(comparison)
            print(f'Comparison saved to: 03 inference/analysis_reports/{os.path.basename(out_path)}')
