"""
Figures for accuracy-test logs produced by 03 inference/infer.py and training
histories produced by 02 training/train_*.py.

Every figure shows all given models side by side. Accuracy figures use the same
numbers as analysis.py (it reuses load_log and summarize):
    1_accuracy_ci.png      accuracy and macro-F1 with segment-bootstrap 95% CIs
    2_confusion.png        row-normalized confusion matrix per model
    3_per_class.png        per-class precision, recall and F1
    4_timeline.png         every window, correct or wrong, segment by segment
    5_recovery_delay.png   mean late-recovery delay per odor
    6_calibration.png      reliability diagram (confidence vs. accuracy)
From training histories (training_history_*.csv):
    training_loss.png    training and validation loss, selected checkpoint marked
From collected data CSVs (with --visualize):
    <odor>.png           every sensor's readings over that odor's segments
    pca_3d.png           every sample in the first three principal components

Usage:
    python plots.py <log1.csv> <log2.csv> ...        # all figures the logs support
    python plots.py -c -r <log1.csv> <log2.csv> ...  # only the selected figures
    python plots.py -l <log1.csv> <log2.csv> ...     # loss figure from each model's
                                                     # newest training history
    python plots.py <training_history_X.csv> ...     # loss figure from given histories
    python plots.py --visualize <data.csv>           # sensor readings per odor, saved to
                                                     # <stage>/figures/<data>_visuals/
Bare file names are looked up in 03 inference/logs (accuracy logs) or
02 training/logs (training histories). PNGs (300 dpi) are written to
03 inference/figures/<model names>_<timestamp>/ and
02 training/figures/<model names>_<timestamp>/, each with a sources.txt.
"""

import argparse
import glob
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # no display needed (Jetson over SSH)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

from analysis import load_log, summarize, _model_name, _resolve

DPI = 300
MIN_DELTA = 1e-4  # val-loss improvement the trainers count as significant
HISTORY_PREFIX = 'training_history_'
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
INFERENCE_DIR = os.path.join(_ROOT, '03 inference')
FIGURES_DIR = os.path.join(INFERENCE_DIR, 'figures')
INFERENCE_CSV_DIR = os.path.join(INFERENCE_DIR, 'csv')
TRAINING_LOGS_DIR = os.path.join(_ROOT, '02 training', 'logs')
TRAINING_FIGURES_DIR = os.path.join(_ROOT, '02 training', 'figures')
TRAINING_CSV_DIR = os.path.join(_ROOT, '02 training', 'csv')

# One fixed color per model across all figures. Unknown models take the next
# free slot, in the order given on the command line.
PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
           '#e87ba4', '#008300', '#4a3aa7', '#e34948']
MODEL_COLORS = {
    'ScentMLPsmall': PALETTE[0],
    'ScentMLProbust': PALETTE[1],
    'ScentCNNsmall': PALETTE[2],
    'ScentCNNrobust': PALETTE[3],
}
GAS_SENSORS = ['NO2_V', 'Eth_V', 'VOC_V', 'CO_V']
ENV_SENSORS = ['Temp', 'Hum']
SENSOR_COLORS = dict(zip(GAS_SENSORS + ENV_SENSORS, PALETTE[:4] + [PALETTE[7], PALETTE[6]]))
ODOR_COLORS = {'air': '#8a8984', 'coffee': '#8c5a2b', 'alcohol': PALETTE[0],
               'lemon': PALETTE[3]}
CORRECT_COLOR = '#d9d8d4' # neutral: correct windows recede
WRONG_COLOR = '#d03b3b'     # errors stand out
NEVER_COLOR = '#e8a0a0'     # faint red hatch: segment never recovered
MISSING_COLOR = '#ffffff'   # window not produced by this model
TEXT_SECONDARY = '#52514e'
GRID_COLOR = '#e4e3df'

plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'axes.edgecolor': '#8a8984',
    'axes.labelcolor': '#0b0b0b',
    'xtick.color': TEXT_SECONDARY,
    'ytick.color': TEXT_SECONDARY,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'savefig.bbox': 'tight',
})


def _colors(names):
    used = set(MODEL_COLORS[n] for n in names if n in MODEL_COLORS)
    spare = [c for c in PALETTE if c not in used]
    out = {}
    for n in names:
        out[n] = MODEL_COLORS[n] if n in MODEL_COLORS else spare.pop(0)
    return out


def _grid(ax, axis='y'):
    ax.grid(axis=axis, color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f'Saved {os.path.relpath(path)}')


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

def plot_accuracy_ci(models, colors, out_dir):
    """Dot plot with 95% CI bars, one row per model, for accuracy and macro-F1."""
    names = list(models)
    y = np.arange(len(names))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 0.5 * len(names) + 1.4), sharey=True)
    for ax, key, title in [(axes[0], 'accuracy', 'Accuracy'),
                           (axes[1], 'macro_f1', 'Macro-F1')]:
        for yi, n in zip(y, names):
            m = models[n][1]
            lo, hi = m['ci'][key]
            ax.errorbar(100 * m[key], yi,
                        xerr=[[100 * (m[key] - lo)], [100 * (hi - m[key])]],
                        fmt='o', color=colors[n], ecolor=colors[n], markersize=7,
                        elinewidth=2, capsize=4)
            ax.annotate(f'{100 * m[key]:.1f}', (100 * m[key], yi), xytext=(0, 7),
                        textcoords='offset points', ha='center', fontsize=8,
                        color=TEXT_SECONDARY)
        ax.set_title(title)
        ax.set_xlabel('%  (dot = observed, bar = 95% CI)')
        _grid(ax, 'x')
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names)
    axes[0].set_ylim(y.min() - 0.6, y.max() + 0.8)
    fig.tight_layout()
    _save(fig, out_dir, '1_accuracy_ci.png')


def plot_confusion(models, colors, classes, out_dir):
    """
    Row-normalized confusion matrix per model, shaded white -> the model's color;
    cells show % of true class and count.
    """
    n = len(models)
    ncols = min(n, 2)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.2 * nrows),
                             squeeze=False)
    for ax, (name, (_, m)) in zip(axes.flat, models.items()):
        cm = m['confusion'].reindex(index=classes, columns=classes, fill_value=0).to_numpy()
        totals = cm.sum(axis=1, keepdims=True)
        pct = np.divide(100.0 * cm, totals, out=np.zeros(cm.shape), where=totals > 0)
        cmap = LinearSegmentedColormap.from_list(name, ['#ffffff', colors[name]])
        ax.imshow(pct, cmap=cmap, vmin=0, vmax=100)
        for i in range(len(classes)):
            for j in range(len(classes)):
                r, g, b, _ = cmap(pct[i, j] / 100)
                dark_cell = 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5
                ax.text(j, i, f'{pct[i, j]:.0f}%\n({cm[i, j]})', ha='center', va='center',
                        fontsize=7.5, color='white' if dark_cell else '#0b0b0b')
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(name)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    for ax in list(axes.flat)[n:]:
        ax.axis('off')
    fig.tight_layout()
    _save(fig, out_dir, '2_confusion.png')


def _grouped_bars(ax, models, colors, classes, values):
    """values[name] -> list aligned with classes (NaN drawn as empty)."""
    names = list(models)
    width = 0.8 / len(names)
    x = np.arange(len(classes))
    for k, n in enumerate(names):
        v = np.nan_to_num(np.asarray(values[n], dtype=float))
        ax.bar(x - 0.4 + width * (k + 0.5), v, width * 0.9, color=colors[n], label=n)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    _grid(ax)


def plot_per_class(models, colors, classes, out_dir):
    """Three panels (precision, recall, F1): odors on x, one bar per model."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, metric in zip(axes, ['Precision', 'Recall', 'F1']):
        values = {}
        for n, (_, m) in models.items():
            pc = m['per_class'].set_index('Class')
            values[n] = [100 * pc[metric].get(c, np.nan) for c in classes]
        _grouped_bars(ax, models, colors, classes, values)
        ax.set_title(metric)
        ax.set_ylim(0, 105)
    axes[0].set_ylabel('%')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(models),
               bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    _save(fig, out_dir, '3_per_class.png')


def plot_timeline(models, out_dir):
    """
    One row per model: segments side by side, each window a cell colored
    correct or wrong. Never-recovered segments are hatched instead.
    """
    names = list(models)
    ref = models[names[0]][1]['segments']
    n_seg = max(len(m['segments']) for _, m in models.values())
    width = max(int(m['segments']['Windows'].max()) for _, m in models.values())
    gap = 1  # blank column between segments

    # 0 = missing or never recovered (hatched below), 1 = correct, 2 = wrong
    grid = np.zeros((len(names), n_seg * (width + gap)))
    never = []  # (row, first column, windows)
    for r, n in enumerate(names):
        df = models[n][0]
        for s, (_, g) in enumerate(df.groupby('Segment')):
            c = g['Correct'].to_numpy()
            start = s * (width + gap)
            if c.any():
                grid[r, start:start + len(c)] = np.where(c, 1, 2)
            else:
                never.append((r, start, len(c)))
    cmap = ListedColormap([MISSING_COLOR, CORRECT_COLOR, WRONG_COLOR])

    fig, ax = plt.subplots(figsize=(11, 0.45 * len(names) + 1.6))
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=2, aspect='auto', interpolation='nearest')
    for r, start, windows in never:
        ax.add_patch(Rectangle((start - 0.5, r - 0.5), windows, 1, facecolor='none',
                               edgecolor=NEVER_COLOR, hatch='////', linewidth=0))
    for r in range(1, len(names)):
        ax.axhline(r - 0.5, color='white', linewidth=3)
    centers = [s * (width + gap) + width / 2 - 0.5 for s in range(n_seg)]
    ax.set_xticks(centers)
    ax.set_xticklabels([f'{i + 1}\n{lab}' for i, lab in enumerate(ref['Label'])],
                       fontsize=7)
    ax.set_xlabel('Segment (true odor)')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(handles=[Patch(color=CORRECT_COLOR, label='Correct window'),
                       Patch(color=WRONG_COLOR, label='Wrong window'),
                       Patch(facecolor='none', edgecolor=NEVER_COLOR, hatch='////',
                             linewidth=0, label='Never-recovered segment')],
              loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.0 + 0.6 / len(names)))
    fig.tight_layout()
    _save(fig, out_dir, '4_timeline.png')


def plot_recovery_delay(models, colors, classes, out_dir):
    """
    Mean late-recovery delay per odor. '-' marks odors with no late-recovered
    segment (all correct at start, or never recovered; see the timeline).
    """
    fig, ax = plt.subplots(figsize=(7, 3.4))
    values = {}
    for n, (_, m) in models.items():
        rc = m['recovery'].set_index('Class')
        values[n] = [rc['Mean delay'].get(c, np.nan) for c in classes]
    _grouped_bars(ax, models, colors, classes, values)

    width = 0.8 / len(models)
    top = np.nanmax([v for vs in values.values() for v in vs] + [1.0])
    for k, n in enumerate(models):
        for i, v in enumerate(values[n]):
            x = i - 0.4 + width * (k + 0.5)
            ax.text(x, (0 if np.isnan(v) else v) + top * 0.02,
                    '-' if np.isnan(v) else f'{v:.2f}', ha='center', va='bottom',
                    fontsize=7, color=TEXT_SECONDARY)
    ax.set_ylim(0, top * 1.15)
    ax.set_ylabel('Mean delay to first\ncorrect window (s)')
    ax.set_title('Recovery delay after odor change (late-recovered segments only)')
    ax.legend(ncol=len(models), loc='upper center', bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    _save(fig, out_dir, '5_recovery_delay.png')


def plot_calibration(models, colors, out_dir):
    """
    Reliability diagram: mean confidence vs. accuracy per bin, one line per model.
    Point area grows with the number of windows in the bin.
    """
    def area(windows):
        return 25 + 0.35 * np.asarray(windows, dtype=float)

    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 100], [0, 100], linestyle='--', color='#8a8984', linewidth=1,
            label='Perfect calibration')
    for n, (_, m) in models.items():
        cal = m['calibration']
        if cal is None:
            continue
        x, y = 100 * cal['Mean confidence'], 100 * cal['Accuracy']
        ax.plot(x, y, '-', color=colors[n], linewidth=1.5,
                label=f"{n} (ECE {100 * m['ece']:.1f}%)")
        ax.scatter(x, y, s=area(cal['Windows']), color=colors[n], linewidths=0, zorder=3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=7.5)
    ax.set_xlim(0, 101)
    ax.set_ylim(0, 101)
    ax.set_aspect('equal')
    ax.set_xlabel('Mean confidence (%)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Calibration')
    _grid(ax, 'both')
    fig.tight_layout()
    _save(fig, out_dir, '6_calibration.png')


def selected_epoch(history):
    """
    Epoch whose weights the trainer restored: the first epoch that improved the
    best validation loss by more than MIN_DELTA, as in train_CNN.py / train_MLP.py.
    """
    best_loss, best_epoch = float('inf'), None
    for epoch, loss in zip(history['epoch'], history['val_loss']):
        if loss < best_loss - MIN_DELTA:
            best_loss, best_epoch = loss, int(epoch)
    return best_epoch


def plot_training_loss(histories, colors, out_dir):
    """
    One panel per model: training loss (dashed) and validation loss (solid) per
    epoch, with the selected checkpoint marked.
    """
    n = len(histories)
    ncols = min(n, 2)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.6 * nrows),
                             squeeze=False)
    for ax, (name, h) in zip(axes.flat, histories.items()):
        best = selected_epoch(h)
        ax.plot(h['epoch'], h['train_loss'], '--', color=colors[name], linewidth=1.5)
        ax.plot(h['epoch'], h['val_loss'], '-', color=colors[name], linewidth=2)
        if best is not None:
            best_loss = float(h.loc[h['epoch'] == best, 'val_loss'].iloc[0])
            ax.axvline(best, color='#8a8984', linestyle=':', linewidth=1)
            ax.plot(best, best_loss, 'o', color='#0b0b0b', markersize=6, zorder=3)
            ax.annotate(f'epoch {best}', (best, 1), xycoords=('data', 'axes fraction'),
                        xytext=(3, -2), textcoords='offset points', ha='left', va='top',
                        fontsize=7.5, color=TEXT_SECONDARY)
        ax.set_title(name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_xlim(0, h['epoch'].max() + 1)
        ax.set_ylim(0, None)
        _grid(ax)
    for ax in list(axes.flat)[n:]:
        ax.axis('off')
    handles = [Line2D([], [], color='#8a8984', linestyle='--', linewidth=1.5,
                      label='Training loss'),
               Line2D([], [], color='#8a8984', linestyle='-', linewidth=2,
                      label='Validation loss'),
               Line2D([], [], color='#0b0b0b', marker='o', linestyle=':',
                      label='Selected checkpoint')]
    fig.tight_layout()
    fig.legend(handles=handles, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.0))
    _save(fig, out_dir, 'training_loss.png')


def _odor_time(odor_df):
    """
    Elapsed seconds with the odor's segments joined back to back: each segment
    continues one sample period after the previous one ends. Returns the time
    array and the start time of every segment after the first.
    """
    ts = odor_df['Timestamp']
    period = ts.diff().dt.total_seconds().median()
    period = 0.25 if np.isnan(period) else period
    t = np.empty(len(odor_df))
    starts, offset, i = [], 0.0, 0
    for _, seg in odor_df.groupby('Segment', sort=True):
        rel = (seg['Timestamp'] - seg['Timestamp'].iloc[0]).dt.total_seconds().to_numpy()
        if i:
            starts.append(offset)
        t[i:i + len(seg)] = offset + rel
        offset += rel[-1] + period
        i += len(seg)
    return t, starts


def plot_sensor_readings(df, odor, out_dir):
    """
    Every sensor of one odor against elapsed time: gas sensors (solid) on the
    left axis in volts, temperature and humidity (dashed) on the right axis.
    Faint vertical lines mark where a new segment of the odor starts.
    """
    odor_df = df[df['Label'] == odor]
    t, starts = _odor_time(odor_df)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    env_ax = ax.twinx()
    env_ax.spines['right'].set_visible(True)
    for s in starts:
        ax.axvline(s, color=GRID_COLOR, linewidth=1, zorder=0)
    lines = []
    for col in GAS_SENSORS:
        lines += ax.plot(t, odor_df[col], '-', color=SENSOR_COLORS[col], linewidth=1.2,
                         label=col)
    for col in ENV_SENSORS:
        lines += env_ax.plot(t, odor_df[col], '--', color=SENSOR_COLORS[col],
                             linewidth=1.2, label=col)
    ax.set_xlim(0, t.max())
    ax.set_xlabel('Elapsed time (s)  (segments joined; vertical lines = new segment)')
    ax.set_ylabel('Sensor voltage (V)')
    env_ax.set_ylabel('Temp (°C) / Hum (%)')
    ax.set_title(f'{odor}  ({len(starts) + 1} segments, {len(odor_df)} samples)')
    _grid(ax)
    ax.legend(handles=lines, loc='upper center', ncol=len(lines),
              bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    _save(fig, out_dir, f'{odor}.png')


def pca(x, n_components=3):
    """
    PCA of the z-scored columns of x (sensors have different units and scales).
    Returns the projected samples and each component's explained-variance ratio.
    """
    std = x.std(axis=0)
    z = (x - x.mean(axis=0)) / np.where(std > 0, std, 1)
    _, s, vt = np.linalg.svd(z, full_matrices=False)
    explained = s ** 2 / np.sum(s ** 2)
    return z @ vt[:n_components].T, explained[:n_components]


def plot_pca_3d(df, out_dir):
    """
    Every sample as a point in the space of the first three principal components
    of the six z-scored sensors, colored by odor.
    """
    proj, explained = pca(df[GAS_SENSORS + ENV_SENSORS].to_numpy(dtype=float))
    odors = list(df['Label'].unique())
    spare = [c for c in PALETTE if c not in ODOR_COLORS.values()]
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection='3d')
    for odor in odors:
        mask = (df['Label'] == odor).to_numpy()
        color = ODOR_COLORS[odor] if odor in ODOR_COLORS else spare.pop(0)
        ax.scatter(*proj[mask].T, s=4, color=color, alpha=0.6, linewidths=0,
                   label=f'{odor} ({mask.sum()})', depthshade=False)
    for i, set_label in enumerate([ax.set_xlabel, ax.set_ylabel, ax.set_zlabel]):
        set_label(f'PC{i + 1} ({100 * explained[i]:.1f}%)')
    ax.set_title(f'PCA of the 6 z-scored sensors  '
                 f'({100 * explained.sum():.1f}% of variance shown)')
    ax.legend(loc='upper left', markerscale=3)
    fig.tight_layout()
    _save(fig, out_dir, 'pca_3d.png')


def _history_model_name(path):
    name = os.path.splitext(os.path.basename(path))[0]
    if name.startswith(HISTORY_PREFIX):
        name = name[len(HISTORY_PREFIX):]
    parts = name.rsplit('_', 2)  # drop trailing _YYYYMMDD_HHMMSS
    return parts[0] if len(parts) == 3 else name


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

# (flag, short flag, help) for each figure, in output order.
ACCURACY_FIGURES = [
    ('accuracy', '-a', 'accuracy and macro-F1 with 95%% CIs'),
    ('confusion', '-c', 'confusion matrices'),
    ('per_class', '-p', 'per-class precision, recall and F1'),
    ('timeline', '-t', 'per-window timeline'),
    ('delay', '-d', 'recovery delay per odor'),
    ('calibration', '-r', 'calibration (reliability diagram)'),
]
LOSS_FIGURE = ('loss', '-l', 'training and validation loss with the selected checkpoint '
                             '(training histories; saved under 02 training/figures)')
FIGURES = ACCURACY_FIGURES + [LOSS_FIGURE]


def _new_out_dir(base, names, paths, figure_texts):
    """Create <base>/<model names>_<timestamp>/ with a sources.txt listing the inputs."""
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = os.path.join(base, '_'.join(list(names) + [stamp]))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'sources.txt'), 'w', encoding='utf-8') as f:
        f.write(f'Figures generated: {stamp}\n\nLogs used:\n')
        f.writelines(f'  {name}: {os.path.basename(path)}\n'
                     for name, path in zip(names, paths))
        f.write('\nFigures:\n')
        f.writelines(f"  {text.replace('%%', '%')}\n" for text in figure_texts)
    return out_dir


def make_figures(log_paths, selected=None):
    """
    Load every accuracy log, compute metrics, and write the selected accuracy
    figures (all when `selected` is empty) to a new folder.
    """
    selected = set(selected or [f for f, _, _ in ACCURACY_FIGURES])
    models, used_logs = {}, []
    for path in log_paths:
        df = load_log(path)
        if df is not None:
            models[_model_name(path)] = (df, summarize(df))
            used_logs.append(path)
    if not models:
        print('Error: no readable logs.')
        return None

    classes = sorted(set().union(*(m['classes'] for _, m in models.values())))
    colors = _colors(list(models))
    out_dir = _new_out_dir(FIGURES_DIR, models, used_logs,
                           [text for name, _, text in ACCURACY_FIGURES if name in selected])

    if 'accuracy' in selected:
        plot_accuracy_ci(models, colors, out_dir)
    if 'confusion' in selected:
        plot_confusion(models, colors, classes, out_dir)
    if 'per_class' in selected:
        plot_per_class(models, colors, classes, out_dir)
    if 'timeline' in selected:
        plot_timeline(models, out_dir)
    if 'delay' in selected:
        plot_recovery_delay(models, colors, classes, out_dir)
    if 'calibration' in selected:
        plot_calibration(models, colors, out_dir)
    return out_dir


def make_training_figure(history_paths):
    """Load training histories and write the loss figure to a new folder."""
    histories, used = {}, []
    for path in history_paths:
        if not os.path.exists(path):
            print(f'Error: Training history not found: {path}')
            continue
        # '#' lines are run notes appended by the trainer.
        histories[_history_model_name(path)] = pd.read_csv(path, comment='#')
        used.append(path)
    if not histories:
        print('Error: no readable training histories.')
        return None
    out_dir = _new_out_dir(TRAINING_FIGURES_DIR, histories, used, [LOSS_FIGURE[2]])
    plot_training_loss(histories, _colors(list(histories)), out_dir)
    return out_dir


def _resolve_data_csv(path):
    """Bare names are looked up in 02 training/csv, then 03 inference/csv."""
    if os.path.exists(path):
        return path
    for d in (TRAINING_CSV_DIR, INFERENCE_CSV_DIR):
        candidate = os.path.join(d, path)
        if os.path.exists(candidate):
            return candidate
    return None


def make_sensor_figures(csv_name):
    """
    One sensor-readings figure per odor of a collected data CSV, written to
    <figures>/<csv name>_visuals/: 03 inference/figures for CSVs under
    03 inference, 02 training/figures otherwise.
    """
    path = _resolve_data_csv(csv_name)
    if path is None:
        print(f'Error: data CSV not found: {csv_name}')
        return None
    df = pd.read_csv(path, parse_dates=['Timestamp'])
    # A new segment starts whenever the label changes.
    df['Segment'] = (df['Label'] != df['Label'].shift()).cumsum()
    inference = os.path.abspath(path).startswith(os.path.abspath(INFERENCE_DIR) + os.sep)
    base = FIGURES_DIR if inference else TRAINING_FIGURES_DIR
    out_dir = os.path.join(base, os.path.splitext(os.path.basename(path))[0] + '_visuals')
    os.makedirs(out_dir, exist_ok=True)
    for odor in df['Label'].unique():
        plot_sensor_readings(df, odor, out_dir)
    plot_pca_3d(df, out_dir)
    return out_dir


def _is_history(path):
    return os.path.basename(path).startswith(HISTORY_PREFIX)


def _resolve_any(path):
    """Bare names: training histories from 02 training/logs, the rest from 03 inference/logs."""
    if os.path.exists(path):
        return path
    return os.path.join(TRAINING_LOGS_DIR, path) if _is_history(path) else _resolve(path)


def _latest_history(model):
    """Newest training history of a model in 02 training/logs, or None."""
    matches = sorted(glob.glob(os.path.join(TRAINING_LOGS_DIR, f'{HISTORY_PREFIX}{model}_*.csv')))
    return matches[-1] if matches else None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('logs', nargs='*',
                        help='accuracy-test log CSV(s) and/or training history CSV(s); bare names '
                             'are looked up in 03 inference/logs or 02 training/logs')
    group = parser.add_argument_group('figures (default: all that the given logs support)')
    for name, short, text in FIGURES:
        group.add_argument(short, '--' + name.replace('_', '-'), dest=name,
                           action='store_true', help=text)
    parser.add_argument('-v', '--visualize', metavar='CSV',
                        help='plot every sensor per odor from a collected data CSV (bare names '
                             'are looked up in 02 training/csv, then 03 inference/csv); saved '
                             'under that stage\'s figures/<CSV name>_visuals')
    args = parser.parse_args()
    if args.visualize:
        out = make_sensor_figures(args.visualize)
        if out:
            print(f'Figures saved to: {os.path.relpath(out, _ROOT)}')
        if not args.logs:
            sys.exit()
    paths = [_resolve_any(p) for p in args.logs]
    accuracy_logs = [p for p in paths if not _is_history(p)]
    history_logs = [p for p in paths if _is_history(p)]
    selected = [name for name, _, _ in ACCURACY_FIGURES if getattr(args, name)]
    any_flag = bool(selected) or args.loss

    if not paths:
        parser.print_help()
        sys.exit()
    if accuracy_logs and (selected or not any_flag):
        out = make_figures(accuracy_logs, selected)
        if out:
            print(f'Figures saved to: 03 inference/figures/{os.path.basename(out)}')
    elif selected:
        print('Error: accuracy figures need accuracy-test logs.')
    if args.loss and not history_logs:
        # No histories given: use each model's newest training history.
        for name in dict.fromkeys(_model_name(p) for p in accuracy_logs):
            found = _latest_history(name)
            if found:
                history_logs.append(found)
            else:
                print(f'Warning: no training history found for {name}.')
    if history_logs and (args.loss or not any_flag):
        out = make_training_figure(history_logs)
        if out:
            print(f'Figures saved to: 02 training/figures/{os.path.basename(out)}')
