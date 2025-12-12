import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, f_oneway, kruskal, mannwhitneyu
import warnings
from math import pi
import random
from sympy.core.random import random

warnings.filterwarnings('ignore')
sns.set(style="whitegrid")
plt.rcParams.update({
    "font.family": "Palatino Linotype",
    "font.size": 12,
})

def check_normality(data_dict):
    rows = []
    for model, values in data_dict.items():
        stat, p = shapiro(values)
        rows.append({"Model": model, "W-statistic": stat, "p-value": p, "Normal": p > 0.05})
    return pd.DataFrame(rows)


def perform_anova_or_kruskal(data_dict, all_normal):
    data_cleaned = [vals for vals in data_dict.values()]
    if all_normal:
        stat, p = f_oneway(*data_cleaned)
        return "ANOVA", stat, p
    else:
        stat, p = kruskal(*data_cleaned)
        return "Kruskal-Wallis", stat, p


def pairwise_tests(data_dict):
    """Mann-Whitney U test pairwise, with Bonferroni correction"""
    models = list(data_dict.keys())
    results = []
    num_comparisons = len(models) * (len(models) - 1) // 2
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1, m2 = models[i], models[j]
            v1, v2 = data_dict[m1], data_dict[m2]
            stat, p = mannwhitneyu(v1, v2, alternative='two-sided')
            corrected_p = p * num_comparisons
            corrected_p = min(corrected_p, 1.0)
            results.append({
                "Model 1": m1,
                "Model 2": m2,
                "U-statistic": stat,
                "p-value": p,
                "corrected p": corrected_p,
                "Significance": (
                    '***' if corrected_p < 0.001 else
                    '**' if corrected_p < 0.01 else
                    '*' if corrected_p < 0.05 else
                    'ns'
                )
            })
    return pd.DataFrame(results)


def plot_performance_radar_comparison(all_data):
    """
    RADAR CHART - Confronto di modelli con MEDIA delle metriche su tutti gli strumenti.

    all_data: dict
        Struttura {instrument_name: {metric_name: [model1_val, model2_val, model3_val]}}
    """
    from math import pi

    # Palette colori
    colors = sns.color_palette("husl", 3)
    model_labels = ["YOLOv8-N", "EfficientNet & CMT-Unet-Large", "EfficientNet & CMT-Unet-Small"]

    # STEP 1: Calcola la MEDIA di ogni metrica su tutti gli strumenti
    # Raccogli tutte le metriche (assumendo che siano le stesse per tutti gli strumenti)
    first_instrument = list(all_data.values())[0]
    metric_names = list(first_instrument.keys())

    # Dizionario per accumulare i valori di ogni metrica
    # Struttura: {metric_name: {model_idx: [list of values across instruments]}}
    accumulated_metrics = {metric: {0: [], 1: [], 2: []} for metric in metric_names}

    # Accumula valori da tutti gli strumenti
    for instrument, metrics in all_data.items():
        for metric_name in metric_names:
            if metric_name in metrics:
                for model_idx in range(3):
                    accumulated_metrics[metric_name][model_idx].append(metrics[metric_name][model_idx])

    # Calcola le medie
    averaged_metrics = {}
    for metric_name in metric_names:
        averaged_metrics[metric_name] = [
            np.mean(accumulated_metrics[metric_name][0]),  # Media model 0
            np.mean(accumulated_metrics[metric_name][1]),  # Media model 1
            np.mean(accumulated_metrics[metric_name][2])  # Media model 2
        ]

    # STEP 2: Crea il radar chart
    n_metrics = len(metric_names)
    angles = [i / n_metrics * 2 * pi for i in range(n_metrics)]
    angles += angles[:1]  # chiusura del cerchio

    # Setup plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Plot per ogni modello
    for model_idx in range(3):
        values = [averaged_metrics[metric][model_idx] for metric in metric_names]
        values += values[:1]  # chiudi il cerchio

        ax.plot(angles, values, linewidth=2.5, linestyle='solid',
                label=model_labels[model_idx], color=colors[model_idx],
                marker='o', markersize=8)
        ax.fill(angles, values, alpha=0.15, color=colors[model_idx])

    # Label delle metriche
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=16, fontweight='bold')

    # Migliora il posizionamento delle etichette in base all'angolo
    for label, angle in zip(ax.get_xticklabels(), angles):
        angle_deg = np.degrees(angle)
        if angle_deg == 0:
            label.set_horizontalalignment('left')  # sposta leggermente a destra
        elif angle_deg == 180:
            label.set_horizontalalignment('right')  # sposta leggermente a sinistra
        else:
            label.set_horizontalalignment('center')

    # Aggiungi griglia
    ax.grid(True, alpha=0.3)

    # Titolo e legenda

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=14)

    plt.tight_layout()
    plt.savefig("plotsall/radar_average_all_instruments.png", dpi=300, bbox_inches="tight")
    plt.close()

    # STEP 3: Stampa statistiche
    print("\n" + "=" * 80)
    print("📊 AVERAGE PERFORMANCE ACROSS ALL INSTRUMENTS")
    print("=" * 80)
    for metric_name in metric_names:
        print(f"\n{metric_name}:")
        for model_idx, model_name in enumerate(model_labels):
            avg_val = averaged_metrics[metric_name][model_idx]
            std_val = np.std(accumulated_metrics[metric_name][model_idx])
            print(f"  {model_name:40s}: {avg_val:.4f} ± {std_val:.4f}")
    print("=" * 80 + "\n")

def plot_scatter_speed_vs_quality(data_dict, speed_metric="Latency_ms", quality_metrics=None):
    """
    SCATTER PLOT - Mostra il trade-off SPEED vs QUALITY
    X: Speed (latency) - MINORE è MEGLIO
    Y: Quality (IoU/Dice/F1) - MAGGIORE è MEGLIO
    """
    if quality_metrics is None:
        quality_metrics = ["IoU", "Dice", "Sensitivity"]

    fig, axes = plt.subplots(1, len(quality_metrics), figsize=(16, 5))
    if len(quality_metrics) == 1:
        axes = [axes]

    colors = sns.color_palette("husl", len(data_dict))

    for idx, quality_metric in enumerate(quality_metrics):
        ax = axes[idx]

        for (model_idx, (model, metrics)) in enumerate(data_dict.items()):
            if speed_metric in metrics and quality_metric in metrics:
                speed_vals = metrics[speed_metric]
                quality_vals = metrics[quality_metric]

                # Media e std
                speed_mean = np.mean(speed_vals)
                quality_mean = np.mean(quality_vals)
                speed_std = np.std(speed_vals)
                quality_std = np.std(quality_vals)

                # Plot punto centrale
                ax.scatter(speed_mean, quality_mean, s=500, alpha=0.7,
                           color=colors[model_idx], edgecolors='black', linewidth=2,
                           label=model, zorder=3)



                # SOLUZIONE: invertiamo l'assegnazione
                ax.errorbar(speed_mean, quality_mean,
                            xerr=speed_std,  # INVERTITO!
                            yerr=quality_std,  # INVERTITO!
                            fmt='none', ecolor=colors[model_idx],
                            alpha=0.5, capsize=5, capthick=2,
                            elinewidth=2, zorder=2)

        ax.set_xlabel(f'Latency (ms)', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{quality_metric}', fontsize=11, fontweight='bold')
        ax.set_title(f'Speed vs {quality_metric}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig("plotsall/scatter_speed_quality.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_heatmap_comparison(data_dict, metrics_list):
    """
    HEATMAP - Mostra tutte le metriche vs tutti i modelli
    Perfetto per una panoramica completa
    """
    # Prepara matrice: RIGHE = metriche, COLONNE = modelli
    summary_data = {}

    # data_dict ha struttura: {metric_name: {model_name: [values]}}
    for metric_name, model_data in data_dict.items():
        row = {}
        for model_name, values in model_data.items():
            # Calcola la media dei valori per questo modello
            row[model_name] = np.mean(values)
        summary_data[metric_name] = row

    df_heatmap = pd.DataFrame(summary_data).T

    # Verifica che il DataFrame non sia vuoto
    if df_heatmap.empty:
        print("⚠️  Warning: No data available for heatmap")
        return

    print("Heatmap data (actual values):")
    print(df_heatmap)

    # Normalizza per riga (ogni metrica ha il suo range)
    df_normalized = df_heatmap.copy()
    for idx in df_normalized.index:
        min_val = df_normalized.loc[idx].min()
        max_val = df_normalized.loc[idx].max()
        range_val = max_val - min_val
        if range_val > 1e-8:  # Evita divisione per zero
            df_normalized.loc[idx] = (df_normalized.loc[idx] - min_val) / range_val
        else:
            df_normalized.loc[idx] = 0.5  # Se tutti i valori sono uguali

    print("\nHeatmap data (normalized):")
    print(df_normalized)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Heatmap normalizzata
    sns.heatmap(df_normalized, annot=True, fmt='.3f', cmap='RdYlGn', center=0.5,
                cbar_kws={'label': 'Performance (Normalized)'}, ax=ax1, linewidths=1)
    ax1.set_title('Normalized Performance Heatmap', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Metrics', fontsize=11)
    ax1.set_xlabel('Models', fontsize=11)

    # Heatmap con valori reali
    sns.heatmap(df_heatmap, annot=True, fmt='.3f', cmap='viridis',
                ax=ax2, linewidths=1, cbar_kws={'label': 'Value'})
    ax2.set_title('Actual Values Heatmap', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Metrics', fontsize=11)
    ax2.set_xlabel('Models', fontsize=11)

    plt.tight_layout()
    plt.savefig("plotsall/heatmap_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Heatmap saved")

def plot_ranking_bars(data_dict, metric_name, ascending=False):
    """
    RANKING BARS - Classifica i modelli dal migliore al peggiore
    Perfetto per evidenziare il vincitore
    """
    means = {model: np.mean(values) for model, values in data_dict.items()}
    sorted_models = sorted(means.items(), key=lambda x: x[1], reverse=(not ascending))

    models = [m[0] for m in sorted_models]
    values = [m[1] for m in sorted_models]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#2ecc71' if i == 0 else '#3498db' if i == 1 else '#e74c3c' if i == len(models) - 1 else '#95a5a6'
              for i in range(len(models))]

    bars = ax.barh(models, values, color=colors, edgecolor='black', linewidth=2)

    # Aggiungi valore sulla barra
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', ha='left', fontsize=11, fontweight='bold')

    # Badge per top 3
    ax.text(max(values) * 0.95, 0.5, '🥇', fontsize=30, ha='center', alpha=0.3)
    if len(models) > 1:
        ax.text(sorted_models[1][1] * 0.95, 1.5, '🥈', fontsize=25, ha='center', alpha=0.3)

    ax.set_xlabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_title(f'Model Ranking: {metric_name}', fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(values) * 1.15)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(f"plotsall/ranking_{metric_name}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_efficiency_frontier(data_dict, speed_metric="Latency_ms", quality_metric="IoU"):
    """
    PARETO FRONTIER - Mostra modelli non-dominati (Efficiency Frontier)
    Un modello è DOMINATO se ne esiste uno che ha MEGLIO sia speed che quality
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Calcola media per ogni modello
    models_data = []
    for model, metrics in data_dict.items():
        if speed_metric in metrics and quality_metric in metrics:
            speed_mean = np.mean(metrics[speed_metric])
            quality_mean = np.mean(metrics[quality_metric])
            models_data.append({
                'model': model,
                'speed': speed_mean,
                'quality': quality_mean
            })

    # Ordina per efficiency
    df_models = pd.DataFrame(models_data)
    df_models = df_models.sort_values('speed')

    # Identifica Pareto frontier
    frontier_mask = []
    for idx, row in df_models.iterrows():
        is_dominated = False
        for idx2, row2 in df_models.iterrows():
            if idx != idx2:
                # Dominato se: speed peggiore E quality peggiore
                if row2['speed'] <= row['speed'] and row2['quality'] >= row['quality']:
                    if not (row2['speed'] == row['speed'] and row2['quality'] == row['quality']):
                        is_dominated = True
                        break
        frontier_mask.append(not is_dominated)

    df_models['frontier'] = frontier_mask

    # Plot
    colors = ['#2ecc71' if f else '#e74c3c' for f in df_models['frontier']]
    sizes = [300 if f else 150 for f in df_models['frontier']]

    ax.scatter(df_models['speed'], df_models['quality'], s=sizes, c=colors,
               alpha=0.7, edgecolors='black', linewidth=2)

    # Linea frontier
    frontier_df = df_models[df_models['frontier']].sort_values('speed')
    if len(frontier_df) > 1:
        ax.plot(frontier_df['speed'], frontier_df['quality'], 'g--', linewidth=2,
                alpha=0.5, label='Pareto Frontier')

    # Etichette
    for idx, row in df_models.iterrows():
        speed_str = f"{row['speed']:.2f}"
        quality_str = f"{row['quality']:.4f}"
        ax.annotate(row['model'] + "(" + speed_str +","+ quality_str+")", (row['speed'], row['quality']),
                    xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')


    # Annotazioni zone
    ax.text(0.98, 0.98, 'BEST\n(Fast & Accurate)', transform=ax.transAxes,
            fontsize=11, ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.3))
    ax.text(0.02, 0.02, 'WORST\n(Slow & Inaccurate)', transform=ax.transAxes,
            fontsize=11, ha='right', va='top', bbox=dict(boxstyle='round', facecolor='#e74c3c', alpha=0.3))

    ax.set_xlabel(f'Latency (ms) ', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{quality_metric}', fontsize=12, fontweight='bold')
    ax.set_title('Efficiency Frontier (Pareto Optimality)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig("plotsall/pareto_frontier.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_violin_comprehensive(data_dict, metrics_list):
    """
    MULTI-METRIC VIOLIN PLOT - Confronta distribuzioni di multiple metriche
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (metric_name, metric_values) in enumerate(metrics_list.items()):
        if idx >= 4:
            break

        ax = axes[idx]

        # Prepara dataframe
        df_plot = pd.DataFrame([
            (model, value) for model, values in metric_values.items() for value in values
        ], columns=["Model", metric_name])

        # Violin + box + media
        sns.violinplot(x="Model", y=metric_name, data=df_plot, ax=ax,
                       palette='Set2', alpha=0.7, inner=None)
        sns.boxplot(x="Model", y=metric_name, data=df_plot, ax=ax,
                    width=0.3, showcaps=False, boxprops={'facecolor': 'none'},
                    medianprops={'color': 'orange', 'linewidth': 2}, showfliers=False)

        x_labels = [label.get_text() for label in ax.get_xticklabels()]

        # Calcola le medie nell'ORDINE CORRETTO del grafico
        means = df_plot.groupby("Model")[metric_name].mean()

        # Plotta le medie nell'ordine corretto
        for i, model_name in enumerate(x_labels):
            if model_name in means:
                mean_val = means[model_name]
                ax.plot(i, mean_val, marker='D', color='red', markersize=12,
                        markeredgecolor='darkred', markeredgewidth=2, zorder=5)

        ax.set_title(f'{metric_name} Distribution', fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=30)

    # Nascondi ultimo se non usato
    if len(metrics_list) < 4:
        axes[-1].set_visible(False)

    plt.suptitle('Comprehensive Metrics Distribution', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig("plotsall/violin_comprehensive.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_speedup_comparison(data_dict, speed_metric="Latency_ms"):
    """
    SPEEDUP BARS - Mostra speedup relativo a un modello baseline (il più lento)
    """
    if speed_metric not in data_dict:
        return

    means = {model: np.mean(values) for model, values in data_dict[speed_metric].items()}
    max_latency = max(means.values())

    speedups = {model: max_latency / latency for model, latency in means.items()}
    sorted_speedups = sorted(speedups.items(), key=lambda x: x[1], reverse=True)

    models = [m[0] for m in sorted_speedups]
    speedup_vals = [m[1] for m in sorted_speedups]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(models))
    bars = ax.bar(models, speedup_vals, color=colors, edgecolor='black', linewidth=2)

    # Linea baseline
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Baseline (Slowest)')

    # Valore sulla barra
    for bar, val in zip(bars, speedup_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{val:.2f}x', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Speedup (relative to slowest)', fontsize=12, fontweight='bold')
    ax.set_title(f'Relative Speedup ({speed_metric})', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(speedup_vals) * 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()

    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig("plotsall/speedup_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_box_with_points_split(data_dict, metric_name, sort_by_value=False):
    """Due grafici separati: uno per modelli veloci, uno per SAM"""
    df = pd.DataFrame([
        (model, value) for model, values in data_dict.items() for value in values
    ], columns=["Model", metric_name])

    means = df.groupby("Model")[metric_name].mean().sort_values(ascending=False)
    model_order = means.index.tolist() if sort_by_value else None
    plot_order = model_order if model_order else df['Model'].unique()

    # Identifica SAM (outlier)
    sam_models = [m for m in plot_order if 'SAM' in m.upper() and m == 'SAM']
    other_models = [m for m in plot_order if m not in sam_models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6),
                                   gridspec_kw={'width_ratios': [len(other_models), len(sam_models)]})
    palette = sns.color_palette("Set2", len(data_dict))

    # Grafico 1: Modelli veloci (senza SAM)
    df_fast = df[df['Model'].isin(other_models)]
    if not df_fast.empty:
        sns.boxplot(x="Model", y=metric_name, data=df_fast, palette=palette,
                    showfliers=False, boxprops=dict(alpha=0.7), width=0.6, ax=ax1,
                    order=other_models)


        for i, model_name in enumerate(other_models):
            if model_name in means:
                ax1.plot(i, means[model_name], marker='D', color='red', markersize=12,
                         markeredgecolor='darkred', markeredgewidth=2, zorder=5)


        ax1.set_ylabel(f'{metric_name} (ms)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.tick_params(axis='x', rotation=30)

    # Grafico 2: SAM
    df_sam = df[df['Model'].isin(sam_models)]
    if not df_sam.empty:
        sns.boxplot(x="Model", y=metric_name, data=df_sam, palette=palette,
                    showfliers=False, boxprops=dict(alpha=0.7), width=0.6, ax=ax2,
                    order=sam_models)


        for i, model_name in enumerate(sam_models):
            if model_name in means:
                ax2.plot(i, means[model_name], marker='D', color='red', markersize=12,
                         markeredgecolor='darkred', markeredgewidth=2, zorder=5)


        ax2.set_ylabel('')
        ax2.set_xlabel('')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(axis='x', rotation=30)

    fig.suptitle(f'{metric_name}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"plotsall/{metric_name}_boxplot.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Stampa medie
    print(f"\n📊 {metric_name} - Means:")
    for model_name in plot_order:
        if model_name in means:
            print(f"  {model_name:20s} = {means[model_name]:8.2f} ms")
def plot_box_with_points(data_dict, metric_name):
    """Versione TESI - Boxplot + punti + media rossa"""
    df = pd.DataFrame([
        (model, value) for model, values in data_dict.items() for value in values
    ], columns=["Model", metric_name])

    fig, ax = plt.subplots(figsize=(10, 7))
    palette = sns.color_palette("Set2", len(data_dict))

    sns.boxplot(x="Model", y=metric_name, data=df, palette=palette,
                showfliers=False, boxprops=dict(alpha=0.7), width=0.6, ax=ax)
   # sns.stripplot(x="Model", y=metric_name, data=df, color="black",
    #              size=7, jitter=True, alpha=0.6, edgecolor="k", linewidth=0.5, ax=ax)
    # Ottieni l'ordine effettivo dei modelli dal grafico
    x_labels = [label.get_text() for label in ax.get_xticklabels()]

    # Calcola le medie nell'ORDINE CORRETTO del grafico
    means = df.groupby("Model")[metric_name].mean()

    # Plotta le medie nell'ordine corretto
    for i, model_name in enumerate(x_labels):
        if model_name in means:
            mean_val = means[model_name]
            ax.plot(i, mean_val, marker='D', color='red', markersize=12,
                    markeredgecolor='darkred', markeredgewidth=2, zorder=5)

    # Stampa le medie nell'ORDINE DEL GRAFICO
    print(f"\n📊 {metric_name} - Means (in plot order):")
    for i, model_name in enumerate(x_labels):
        if model_name in means:
            print(f"  {i} ('{model_name}', {means[model_name]:.16f})")

    ax.set_title(f'{metric_name}', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel(metric_name, fontsize=13)
    ax.set_xlabel('')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()

    plt.savefig(f"plotsall/{metric_name}_boxplot.png", dpi=300, bbox_inches='tight')
    plt.close()
# ============================================
# MAIN - Genera TUTTI i grafici
# ============================================

def main(file_path="results.xlsx"):
    os.makedirs("plotsall", exist_ok=True)

    xls = pd.ExcelFile(file_path)
    metrics = xls.sheet_names
    instruments = xls.parse(metrics[0],usecols=[0]).dropna(axis=1,how='all')
    instruments = instruments.iloc[:,0].dropna().tolist()
    print(instruments)

    # Leggi tutti i dati
    all_data = {}
    for metric in metrics:
        for instrument in instruments:
            df = xls.parse(metric).dropna(axis=1, how='all')

            df = df[df.iloc[:, 0 ]== instrument]

            if instrument not in all_data:
                all_data[instrument] = {}


            all_data[instrument][metric] = df.iloc[0,1:].dropna().astype(float).tolist() #dizionario con chiave strumento, poi metrica con lista di valori in oridne yolo,large,small

    print(all_data)
    print("📊 Generating presentation plots...")


    plot_performance_radar_comparison(
        all_data,

    )
    print(f"  ✓ Performance radar - Custom selection")

    print("\n✅ All presentation plots saved to 'plotsall/' folder!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="results.xlsx")
    args = parser.parse_args()

    main(args.input_file)