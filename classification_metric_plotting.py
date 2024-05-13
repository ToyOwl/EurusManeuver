import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import argparse

def read_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    return df

def smooth_values(y, window_size, poly_order):
    if window_size % 2 == 0:
        window_size += 1
    return savgol_filter(y, window_size, poly_order)

def plot_multiple_data(dfs, smooth=False, window_size=5, poly_order=3):
    plt.figure(figsize=(10, 6))

    for df in dfs:
        grouped = df.groupby('tag')
        for name, group in grouped:
            x = group['id']
            y = group['value']
            if smooth:
                y = smooth_values(y, window_size, poly_order)
            plt.plot(x, y, label=name)

    plt.xlabel("iter")
    plt.ylabel("value")
    plt.legend()
    plt.show()


def plot_train_valid_data(dfs, smooth=False, window_length=5, poly_order=3, iter_type="batch"):
  colors = {"train": "#1f77b4", "valid": "#ff7f0e"}
  line_styles = {"train": "-", "valid": "--"}
  labels = {"train": "Train", "valid": "Valid"}

  metrics = set(tag.split("/")[0] for df in dfs for tag in df["tag"])
  setups = set(tag.split("/")[1] for df in dfs for tag in df["tag"])

  plt.figure(figsize=(10, 6))
  ax = plt.gca()
  ax.set_facecolor("#f8f8f8")
  plt.grid(True, which="both", linestyle="--", linewidth=0.5, color="gray", alpha=0.5)

  for metric in metrics:
     for df in dfs:
        df_filtered = df[df["tag"].str.contains(metric)]
        for setup in setups:
           data = df_filtered[df_filtered["tag"].str.contains(setup)]
           if not data.empty:
             x = data["id"]
             y = data["value"]
             if smooth and len(y) > window_length:
               y_smoothed = savgol_filter(y, window_length, poly_order)
               plt.plot(x, y_smoothed, label=f"{metric.capitalize()} ({labels[setup]})",
                                 color=colors[setup], linestyle=line_styles[setup], linewidth=2)
               plt.plot(x, y, color=colors[setup], linestyle=line_styles[setup], linewidth=1, alpha=0.3)
             else:
               plt.plot(x, y, label=f"{metric.capitalize()} ({labels[setup]})",
                                  color=colors[setup], linestyle=line_styles[setup], linewidth=2)


     title = " vs ".join(sorted(metrics)) if len(metrics) > 1 else f"{metric.capitalize()} Metric Evaluation"
     plt.title(title, fontsize=14, fontweight='bold')
     plt.xlabel(iter_type, fontsize=12)
     plt.legend(frameon=True, framealpha=0.9, facecolor='#f0f0f0')
     plt.xticks(fontsize=10)
     plt.yticks(fontsize=10)
     plt.tight_layout()

  plt.show()

def main():
  parser = argparse.ArgumentParser(description="Plot multiple functions from CSV data")
  parser.add_argument("--csv_files", nargs="+", help="Paths to the CSV files")
  parser.add_argument("--smooth", action="store_true", help="Apply smoothing to the graph lines")
  parser.add_argument("--window_size", type=int, default=5, help="Window size for the smoothing filter")
  parser.add_argument("--order", type=int, default=3, help="Polynomial order for the smoothing filter")
  args = parser.parse_args()

  dfs = [read_and_prepare_data(file) for file in args.csv_files]

  plot_train_valid_data(dfs, smooth=args.smooth, window_length=args.window_size, poly_order=args.order)



if __name__ == '__main__':
    main()

