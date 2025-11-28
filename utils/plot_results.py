import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os


def plot_mim_curve(csv_path, output_dir):
    if not os.path.exists(csv_path):
        print(f"⚠️ Warning: {csv_path} not found yet. Wait for training to finish.")
        return

    # Load Data
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Set Style (Academic Paper Standard)
    sns.set_theme(style="whitegrid", context="paper")
    plt.figure(figsize=(8, 5))

    # Plot Loss
    sns.lineplot(data=df, x='epoch', y='loss', linewidth=2.5, color='#E63946')

    # Labels & Title
    plt.title("MIM Pre-training Convergence (Self-Supervised)", fontsize=12, fontweight='bold')
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("Reconstruction MSE Loss", fontsize=10)
    plt.xlim(0, df['epoch'].max() + 1)

    # Save
    out_file = os.path.join(output_dir, "training_curve.png")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    print(f"✅ Plot saved to {out_file}")


if __name__ == "__main__":
    # Default to the experiment 1 folder
    DEFAULT_CSV = "runs/mim_experiment_1/metrics.csv"
    DEFAULT_OUT = "runs/mim_experiment_1"

    plot_mim_curve(DEFAULT_CSV, DEFAULT_OUT)