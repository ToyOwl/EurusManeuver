import argparse
import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def preprocess_data(df):
  cuda_models = df[df["Device"] == "cuda"]["Model"].unique()
  for model in cuda_models:
      cuda_accuracy = df[(df["Model"] == model) & (df["Device"] == 'cuda')]["Accuracy"].values
      if cuda_accuracy.size > 0:
         df.loc[(df["Model"] == model) & (df["Device"] == "cpu"), "Accuracy"] = cuda_accuracy

  return df

def plot_model_inference_metric(df, x_metric, y_metric, output_dir):

  x_normalized = df[x_metric] / df[x_metric].max() * 1000

  min_size = 50
  x_normalized = np.maximum(x_normalized, min_size)

  norm = plt.Normalize(df[x_metric].min(), df[x_metric].max())

  plt.figure(figsize=(12, 8))
  ax = plt.gca()
  ax.set_facecolor("#f9f9f9")
  cmap = plt.get_cmap("plasma")

  scatter = plt.scatter(df[x_metric], df[y_metric], s=x_normalized, norm=norm, c=df[x_metric],
                        alpha=0.6, edgecolors="w", linewidth=0.6)

  for i, row in df.iterrows():
     plt.annotate(row["Model"], (row[x_metric], row[y_metric]),
                  textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color="dimgrey")

  plt.title(f"{x_metric} vs. {y_metric}", fontsize=16, fontweight="bold", color="dimgrey")
  plt.xlabel(x_metric, fontsize=12, fontweight="bold", color="dimgrey")
  plt.ylabel(y_metric, fontsize=12, fontweight="bold", color="dimgrey")
  plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey', alpha=0.3)

  sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(df[x_metric]), vmax=max(df[x_metric])))
  cbar = plt.colorbar(scatter)
  cbar.set_label(x_metric, fontsize=12)
  cbar.ax.tick_params(labelsize=10)

  plt.xticks(fontsize=10)
  plt.yticks(fontsize=10)
  plt.tight_layout()

  if not os.path.exists(output_dir):
      os.makedirs(output_dir)

  file_path = os.path.join(output_dir, f"{x_metric}_vs_{y_metric}.png")
  plt.savefig(file_path)
  plt.close()
  print(f"Plot saved to {file_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot performance metrics for different models")
    parser.add_argument("--csv_path", help="Path to the CSV file containing MMACS, accuracy, and model-name columns")
    parser.add_argument("--smooth", action="store_true", help="Apply a smoothing effect to the circle sizes")
    parser.add_argument("--output_dir", default="reports", help="Directory to save output images.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    mod_df = preprocess_data(df)


    plot_model_inference_metric(mod_df, "Params (M)", "Accuracy", args.output_dir)

    cpu_df = mod_df[mod_df["Device"] == "cpu"]
    plot_model_inference_metric(cpu_df, "Latency (ms)", "Accuracy", args.output_dir)

if __name__ == "__main__":
    main()
