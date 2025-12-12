import torch
import numpy as np
import cv2
import os
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from collections import defaultdict
import json


class YOLOStyleReporter:
    """
    Genera report in stile YOLO per classificazione multi-classe.
    """

    def __init__(self, class_names, save_dir='resultsMultiClass'):
        """
        Args:
            class_names: Dict {class_id: 'class_name'}
            save_dir: Directory per salvare i report
        """
        self.class_names = class_names
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Statistiche per classe
        self.stats = {
            'predictions': [],
            'ground_truths': [],
            'confidences': [],
            'processing_times': [],
            'per_class': defaultdict(lambda: {
                'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0,
                'predictions': [],
                'ground_truths': [],
                'confidences': []
            })
        }

    def update(self, predictions, ground_truths, confidences=None, processing_time=None):
        """
        Aggiorna statistiche con nuove predizioni.
        """
        self.stats['predictions'].extend(predictions)
        self.stats['ground_truths'].extend(ground_truths)

        if confidences is not None:
            self.stats['confidences'].extend(confidences)

        if processing_time is not None:
            self.stats['processing_times'].append(processing_time)

        # Aggiorna statistiche per classe
        for pred, gt in zip(predictions, ground_truths):
            for class_id in self.class_names.keys():
                if pred == class_id and gt == class_id:
                    self.stats['per_class'][class_id]['tp'] += 1
                elif pred == class_id and gt != class_id:
                    self.stats['per_class'][class_id]['fp'] += 1
                elif pred != class_id and gt == class_id:
                    self.stats['per_class'][class_id]['fn'] += 1
                else:
                    self.stats['per_class'][class_id]['tn'] += 1

            self.stats['per_class'][pred]['predictions'].append(pred)
            self.stats['per_class'][gt]['ground_truths'].append(gt)

    def compute_metrics(self):
        """
        Calcola metriche complete.
        """
        preds = np.array(self.stats['predictions'])
        gts = np.array(self.stats['ground_truths'])

        # Metriche globali
        overall_accuracy = np.mean(preds == gts)

        # Metriche per classe
        precision, recall, f1, support = precision_recall_fscore_support(
            gts, preds, labels=list(self.class_names.keys()), zero_division=0
        )

        # Costruisci dizionario metriche
        metrics = {
            'overall': {
                'accuracy': overall_accuracy,
                'total_samples': len(preds),
                'avg_time_ms': np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0
            },
            'per_class': {}
        }

        for idx, class_id in enumerate(self.class_names.keys()):
            tp = self.stats['per_class'][class_id]['tp']
            fp = self.stats['per_class'][class_id]['fp']
            fn = self.stats['per_class'][class_id]['fn']
            tn = self.stats['per_class'][class_id]['tn']

            # AGGIUNTO: Accuracy per classe
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

            metrics['per_class'][class_id] = {
                'name': self.class_names[class_id],
                'accuracy': accuracy,  # NUOVO!
                'precision': precision[idx],
                'recall': recall[idx],
                'f1': f1[idx],
                'support' : sum(1 for gt in gts if gt == class_id),
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
            }

        return metrics

    def generate_pr_curves(self):
        """
        Genera Precision-Recall curves per ogni classe.
        """
        metrics = self.compute_metrics()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Bar plot: Precision, Recall, F1, Accuracy per classe
        classes = []
        accuracies = []
        precisions = []
        recalls = []
        f1s = []

        for class_id, class_metrics in metrics['per_class'].items():
            classes.append(class_metrics['name'])
            accuracies.append(class_metrics['accuracy'])  # NUOVO!
            precisions.append(class_metrics['precision'])
            recalls.append(class_metrics['recall'])
            f1s.append(class_metrics['f1'])

        x = np.arange(len(classes))
        width = 0.2  # Ridotto per fare spazio a 4 barre

        ax1.bar(x - 1.5 * width, accuracies, width, label='Accuracy', color='#9b59b6')  # NUOVO!
        ax1.bar(x - 0.5 * width, precisions, width, label='Precision', color='#2ecc71')
        ax1.bar(x + 0.5 * width, recalls, width, label='Recall', color='#3498db')
        ax1.bar(x + 1.5 * width, f1s, width, label='F1-Score', color='#e74c3c')

        ax1.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax1.set_title('Metrics per Class', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(classes, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim([0, 1.1])

        # Plot 2: Support per classe
        supports = [metrics['per_class'][cid]['support'] for cid in sorted(metrics['per_class'].keys())]

        ax2.bar(classes, supports, color='#f39c12')
        ax2.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Instances', fontsize=12, fontweight='bold')
        ax2.set_title('Support per Class', fontsize=14, fontweight='bold')
        ax2.set_xticklabels(classes, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'metrics_per_class.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def generate_results_table(self):
        """
        Genera tabella risultati in stile YOLO.
        """
        metrics = self.compute_metrics()

        # Crea DataFrame
        data = []
        for class_id in sorted(metrics['per_class'].keys()):
            cm = metrics['per_class'][class_id]
            data.append({
                'Class': cm['name'],
                'Instances': cm['support'],
                'Accuracy': f"{cm['accuracy']:.3f}",  # NUOVO!
                'Precision': f"{cm['precision']:.3f}",
                'Recall': f"{cm['recall']:.3f}",
                'F1-Score': f"{cm['f1']:.3f}",
                'Specificity': f"{cm['specificity']:.3f}"
            })

        df = pd.DataFrame(data)

        # Aggiungi riga totale
        total_row = {
            'Class': 'all',
            'Instances': len(self.stats['predictions']),
            'Accuracy': f"{np.mean([metrics['per_class'][c]['accuracy'] for c in metrics['per_class']]):.3f}",
            'Precision': f"{np.mean([metrics['per_class'][c]['precision'] for c in metrics['per_class']]):.3f}",
            'Recall': f"{np.mean([metrics['per_class'][c]['recall'] for c in metrics['per_class']]):.3f}",
            'F1-Score': f"{np.mean([metrics['per_class'][c]['f1'] for c in metrics['per_class']]):.3f}",
            'Specificity': f"{np.mean([metrics['per_class'][c]['specificity'] for c in metrics['per_class']]):.3f}"
        }
        df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

        # Salva CSV
        df.to_csv(os.path.join(self.save_dir, 'results.csv'), index=False)

        # Crea immagine tabella
        fig, ax = plt.subplots(figsize=(16, len(df) * 0.5 + 1))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='center',
            loc='center',
            colColours=['#f0f0f0'] * len(df.columns)
        )

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Stile header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Evidenzia ultima riga (totale)
        for i in range(len(df.columns)):
            table[(len(df), i)].set_facecolor('#e8f8f5')
            table[(len(df), i)].set_text_props(weight='bold')

        plt.savefig(os.path.join(self.save_dir, 'results_table.png'), dpi=300, bbox_inches='tight')
        plt.close()

        return df

    def generate_summary_plot(self):
        """
        Genera plot riassuntivo in stile YOLO.
        """
        metrics = self.compute_metrics()

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Overall Accuracy
        ax1 = fig.add_subplot(gs[0, :])
        ax1.text(
            0.5, 0.5,
            f"Overall Accuracy: {metrics['overall']['accuracy'] * 100:.2f}%",
            ha='center', va='center',
            fontsize=24, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.8, edgecolor='black', linewidth=2)
        )
        ax1.axis('off')

        # 2. Metriche medie (AGGIUNTA ACCURACY)
        ax2 = fig.add_subplot(gs[1, 0])
        avg_metrics = {
            'Accuracy': np.mean([m['accuracy'] for m in metrics['per_class'].values()]),
            'Precision': np.mean([m['precision'] for m in metrics['per_class'].values()]),
            'Recall': np.mean([m['recall'] for m in metrics['per_class'].values()]),
            'F1-Score': np.mean([m['f1'] for m in metrics['per_class'].values()])
        }
        colors_bar = ['#9b59b6', '#2ecc71', '#3498db', '#e74c3c']
        ax2.bar(avg_metrics.keys(), avg_metrics.values(), color=colors_bar)
        ax2.set_title('Average Metrics', fontweight='bold', fontsize=12)
        ax2.set_ylim([0, 1])
        ax2.grid(axis='y', alpha=0.3)
        ax2.tick_params(axis='x', rotation=15)
        for i, (k, v) in enumerate(avg_metrics.items()):
            ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)

        # 3. Distribuzione classi
        ax3 = fig.add_subplot(gs[1, 1])
        class_counts = [metrics['per_class'][cid]['support'] for cid in sorted(metrics['per_class'].keys())]
        class_labels = [self.class_names[cid] for cid in sorted(metrics['per_class'].keys())]
        ax3.pie(class_counts, labels=class_labels, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Class Distribution', fontweight='bold', fontsize=12)

        # 4. Tempi di elaborazione
        ax4 = fig.add_subplot(gs[1, 2])
        if self.stats['processing_times']:
            times = self.stats['processing_times']
            ax4.hist(times, bins=20, color='#9b59b6', alpha=0.7, edgecolor='black')
            ax4.axvline(np.mean(times), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(times):.1f}ms')
            ax4.set_xlabel('Time (ms)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Processing Time Distribution', fontweight='bold', fontsize=12)
            ax4.legend()
            ax4.grid(alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No timing data', ha='center', va='center', fontsize=12)
            ax4.axis('off')

        # 5. Accuracy per classe (ordinato)
        ax5 = fig.add_subplot(gs[2, :])
        sorted_classes = sorted(
            metrics['per_class'].items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )

        class_names_sorted = [self.class_names[cid] for cid, _ in sorted_classes]
        accuracy_scores = [m['accuracy'] for _, m in sorted_classes]

        colors_perf = ['#2ecc71' if acc > 0.9 else '#f39c12' if acc > 0.7 else '#e74c3c' for acc in accuracy_scores]

        ax5.barh(class_names_sorted, accuracy_scores, color=colors_perf, edgecolor='black')
        ax5.set_xlabel('Accuracy', fontweight='bold')
        ax5.set_title('Accuracy per Class (Sorted)', fontweight='bold', fontsize=12)
        ax5.set_xlim([0, 1])
        ax5.grid(axis='x', alpha=0.3)

        for i, (name, score) in enumerate(zip(class_names_sorted, accuracy_scores)):
            ax5.text(score + 0.02, i, f'{score:.3f}', va='center', fontweight='bold')

        plt.suptitle('Classification Report Summary', fontsize=18, fontweight='bold', y=0.98)
        plt.savefig(os.path.join(self.save_dir, 'summary_report.png'), dpi=300, bbox_inches='tight')
        plt.close()


    def generate_confusion_matrix(self):
        """
        Genera confusion matrix dettagliata.
        """
        preds = self.stats['predictions']
        gts = self.stats['ground_truths']

        labels = sorted(self.class_names.keys())
        label_names = [self.class_names[i] for i in labels]

        cm = confusion_matrix(gts, preds, labels=labels)

        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(12, 10))

        # Normalizza per riga (recall)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)  # Gestisci divisione per zero

        # Heatmap
        sns.heatmap(
            cm_norm,
            annot=cm,  # Mostra valori assoluti
            fmt='d',
            cmap='Blues',
            xticklabels=label_names,
            yticklabels=label_names,
            cbar_kws={'label': 'Normalized Count'},
            ax=ax,
            vmin=0,
            vmax=1
        )

        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('Ground Truth', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix (Normalized by True Label)', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Salva anche versione non normalizzata
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='YlOrRd',
            xticklabels=label_names,
            yticklabels=label_names,
            ax=ax
        )
        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('Ground Truth', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix (Absolute Counts)', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'confusion_matrix_absolute.png'), dpi=300, bbox_inches='tight')
        plt.close()

        return cm


    def save_json_report(self):
        """
        Salva report completo in JSON.
        """
        metrics = self.compute_metrics()

        report = {
            'overall_metrics': metrics['overall'],
            'per_class_metrics': {
                self.class_names[cid]: cm
                for cid, cm in metrics['per_class'].items()
            },
            'confusion_matrix': confusion_matrix(
                self.stats['ground_truths'],
                self.stats['predictions'],
                labels=sorted(self.class_names.keys())
            ).tolist()
        }

        with open(os.path.join(self.save_dir, 'report.json'), 'w') as f:
            json.dump(report, f, indent=2)

    def generate_all_reports(self):
        """
        Genera tutti i report.
        """
        print("\n" + "=" * 70)
        print("📊 GENERAZIONE REPORT")
        print("=" * 70)

        print("\n1️⃣ Confusion Matrix...")
        self.generate_confusion_matrix()

        print("2️⃣ Precision-Recall Curves...")
        self.generate_pr_curves()

        print("3️⃣ Results Table...")
        df = self.generate_results_table()

        print("4️⃣ Summary Plot...")
        self.generate_summary_plot()

        print("5️⃣ JSON Report...")
        self.save_json_report()

        print("\n✅ Report completati!")
        print(f"📁 Salvati in: {self.save_dir}")

        # Stampa tabella su console
        print("\n" + "=" * 70)
        print("RESULTS TABLE")
        print("=" * 70)
        print(df.to_string(index=False))
        print("=" * 70)

        return df


# ============= INTEGRAZIONE NEL TUO CODICE =============

def add_reporting_to_your_code():
    """
    Esempio di come integrare nel tuo codice.
    """

    print("""
# ===== AGGIUNGI ALL'INIZIO DEL TUO SCRIPT =====

from yolo_reports import YOLOStyleReporter

# Inizializza reporter
reporter = YOLOStyleReporter(
    class_names=class_names,  # Il tuo dizionario class_names
    save_dir='results_classification1'
)

# ===== NEL LOOP DI INFERENZA (dopo aver processato un'immagine) =====

# Dopo aver raccolto image_preds e image_labels per un'immagine:
if len(image_preds) > 0:
    reporter.update(
        predictions=image_preds,
        ground_truths=image_labels,
        confidences=None,  # Aggiungi se hai confidence scores
        processing_time=processing_time
    )

# ===== ALLA FINE, GENERA TUTTI I REPORT =====

# Genera tutti i report
reporter.generate_all_reports()

# Salva anche il tuo DataFrame originale
timeDf.to_csv(os.path.join(out_dir, "detection_results.csv"), index=False)
    """)


if __name__ == "__main__":
    add_reporting_to_your_code()

    print("\n" + "=" * 70)
    print("📚 REPORT GENERATI:")
    print("=" * 70)
    print("  ✓ confusion_matrix.png          - Matrice confusione normalizzata")
    print("  ✓ confusion_matrix_absolute.png - Matrice confusione valori assoluti")
    print("  ✓ metrics_per_class.png         - Precision/Recall/F1 per classe")
    print("  ✓ results_table.png              - Tabella risultati completa")
    print("  ✓ summary_report.png             - Report riassuntivo")
    print("  ✓ results.csv                    - Tabella metriche CSV")
    print("  ✓ report.json                    - Report completo JSON")
    print("=" * 70)