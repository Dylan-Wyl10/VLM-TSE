import os
import json
import re
import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class EvaluationTool:
    def __init__(self, txt_path, save_dir="./eval_results"):
        self.txt_path = txt_path
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.cases = []
        self.error_records = []

    def parse_cases(self):
        """
        Parse txt file and extract prediction JSON, groundtruth dict, and missing sensors per case.
        """
        with open(self.txt_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        raw_cases = raw_text.split('-----------------------------------------------------------------------------------------------output:')
        for case_text in raw_cases[1:]:  # Skip first empty
            try:
                # Extract ASSISTANT output (model prediction)
                pred_json_match = re.search(r'ASSISTANT:\s*(```json)?\s*({.*?})\s*(ground truth:)', case_text, re.DOTALL)
                pred_text = pred_json_match.group(2) if pred_json_match else None

                # Extract groundtruth
                gt_match = re.search(r'ground truth:\s*({.*})', case_text, re.DOTALL)
                gt_text = gt_match.group(1) if gt_match else None

                # Extract videos mentioned (missing sensors)
                video_ids = []
                video_matches = re.findall(r'video.*?([Pp]\d+)', case_text)
                for vid in video_matches:
                    sensor_id = int(re.findall(r'\d+', vid)[0])
                    video_ids.append(sensor_id)

                if pred_text and gt_text:
                    pred = json.loads(pred_text)
                    # Handle np.float64 manually
                    gt_text = re.sub(r'np\.float64\((.*?)\)', r'\1', gt_text)
                    gt = ast.literal_eval(gt_text)

                    gt_clean = {
                        "density": {int(k): float(v) for k, v in gt['density'].items()},
                        "speed":   {int(k): float(v) for k, v in gt['speed'].items()},
                        "flow":    {int(k): float(v) for k, v in gt['rate'].items()},
                    }
                    pred_clean = {
                        "density": {int(k): float(v) for k, v in pred['density'].items()},
                        "speed":   {int(k): float(v) for k, v in pred['speed'].items()},
                        "flow":    {int(k): float(v) for k, v in pred['flow'].items()},
                    }

                    self.cases.append({
                        "pred": pred_clean,
                        "gt": gt_clean,
                        "missing_sensors": set(video_ids)
                    })
            except Exception as e:
                print(f"Error parsing a case: {e}")
                continue

    def compute_metrics(self, pred, gt):
        """
        Compute MAE and RMSE between pred and gt for each dimension.
        """
        results = {}
        for key in ["density", "speed", "flow"]:
            pred_vals = np.array([pred[key][sid] for sid in sorted(gt[key].keys())])
            gt_vals = np.array([gt[key][sid] for sid in sorted(gt[key].keys())])

            mae = np.mean(np.abs(pred_vals - gt_vals))
            rmse = np.sqrt(np.mean((pred_vals - gt_vals) ** 2))
            results[key] = {"MAE": mae, "RMSE": rmse}
        return results

    def plot_comparison(self, pred, gt, missing_sensors, case_idx):
        """
        Plot and save comparison figure for one case, marking missing sensors.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        dimensions = ["density", "speed", "flow"]

        for i, key in enumerate(dimensions):
            ax = axes[i]
            sensor_ids = sorted(gt[key].keys())
            gt_vals = [gt[key][sid] for sid in sensor_ids]
            pred_vals = [pred[key][sid] for sid in sensor_ids]

            for sid, gt_val, pred_val in zip(sensor_ids, gt_vals, pred_vals):
                if sid in missing_sensors:
                    ax.scatter(sid, gt_val, color="blue", marker="x", label="GT (video)" if i == 0 else "", s=80)
                    ax.scatter(sid, pred_val, color="red", marker="x", label="Pred (video)" if i == 0 else "", s=80)
                else:
                    ax.scatter(sid, gt_val, color="blue", marker="o", label="GT" if i == 0 else "", s=80)
                    ax.scatter(sid, pred_val, color="red", marker="o", label="Pred" if i == 0 else "", s=80)

            ax.set_title(f"{key.capitalize()}")
            ax.set_xlabel("Sensor ID")
            ax.set_ylabel(key.capitalize())
            ax.grid(True)

        handles, labels = axes[0].get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        fig.legend(unique.values(), unique.keys(), loc="upper right")

        plt.suptitle(f"Case {case_idx} - Prediction vs Groundtruth", fontsize=16)
        save_path = os.path.join(self.save_dir, f"case_{case_idx}_comparison.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_error_heatmap(self, all_errors, error_type="density"):
        """
        Plot and save heatmap of errors across all cases.
        """
        error_matrix = np.stack([case[error_type] for case in all_errors])
        plt.figure(figsize=(12, 6))
        sns.heatmap(error_matrix, cmap="viridis", cbar=True)
        plt.title(f"{error_type.capitalize()} Absolute Error Heatmap")
        plt.xlabel("Sensor ID")
        plt.ylabel("Case Index")
        save_path = os.path.join(self.save_dir, f"{error_type}_error_heatmap.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def evaluate_all(self):
        """
        Run full evaluation: compute metrics, plot comparison, plot heatmaps.
        """
        mae_summary = {"density": [], "speed": [], "flow": []}
        rmse_summary = {"density": [], "speed": [], "flow": []}
        all_errors = []

        for i, case in enumerate(self.cases):
            pred = case["pred"]
            gt = case["gt"]
            missing_sensors = case["missing_sensors"]

            metrics = self.compute_metrics(pred, gt)
            for key in ["density", "speed", "flow"]:
                mae_summary[key].append(metrics[key]["MAE"])
                rmse_summary[key].append(metrics[key]["RMSE"])

            # Record absolute error for heatmap
            single_case_error = {}
            for key in ["density", "speed", "flow"]:
                sensor_ids = sorted(gt[key].keys())
                pred_vals = np.array([pred[key][sid] for sid in sensor_ids])
                gt_vals = np.array([gt[key][sid] for sid in sensor_ids])
                abs_error = np.abs(pred_vals - gt_vals)
                single_case_error[key] = abs_error
            all_errors.append(single_case_error)

            # Plot and save comparison figure
            self.plot_comparison(pred, gt, missing_sensors, case_idx=i)

        # Print average metrics
        print("\n=== Overall Metrics Summary ===")
        for key in ["density", "speed", "flow"]:
            avg_mae = np.mean(mae_summary[key])
            avg_rmse = np.mean(rmse_summary[key])
            print(f"{key.capitalize()} - Avg MAE: {avg_mae:.2f}, Avg RMSE: {avg_rmse:.2f}")

        # Plot heatmaps
        for key in ["density", "speed", "flow"]:
            self.plot_error_heatmap([{key: case[key]} for case in all_errors], error_type=key)

evaluator = EvaluationTool(txt_path="logs_1min.txt", save_dir="../result/eval_results1min")
evaluator.parse_cases()
evaluator.evaluate_all()