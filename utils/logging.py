import cv2
import numpy as np
def log_metrics(log_writer, prefix, idx, loss, recall, accuracy, fpr, fnr):
    log_writer.add_scalar(tag=f"loss/{prefix}", value=loss, step=idx)
    log_writer.add_scalar(tag=f"recall/{prefix}", value=recall, step=idx)
    log_writer.add_scalar(tag=f"accuracy/{prefix}", value=accuracy, step=idx)
    log_writer.add_scalar(tag=f"FPR/{prefix}", value=fpr, step=idx)
    log_writer.add_scalar(tag=f"FNR/{prefix}", value=fnr, step=idx)

def visualize_samples(log_writer, step, sources, targets, predictions, max_num_viz=12):
    indices = np.random.choice(targets.shape[0], min(targets.shape[0], max_num_viz), replace=False)
    for idx in indices:
        img = cv2.normalize(sources[idx, :, :, :], None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        feature_str = "Match" if targets[idx] == predictions[idx] else "Mismatch"
        cv2.putText(img, feature_str, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 0, 0) if 'Mismatch' in feature_str else (0, 255, 0))
        log_writer.add_image(f"Sample {idx}", img, step, dataformats="HWC")