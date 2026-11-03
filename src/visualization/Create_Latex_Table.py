import os
import re

# Directory containing the metrics txt files
metrics_dir = os.path.join("data", "output", "linear")

# Map file suffixes to Model Name, Correction, and Split for ALL models
models_config = [
    # Primary Models
    ("full_corr", "Glint", "train_glint_full_corr.txt", "val_glint_full_corr.txt"),
    ("full_corr", "No-Glint", "train_noglint_full_corr.txt", "val_noglint_full_corr.txt"),
    ("ag_corr", "Glint", "train_glint_ag_corr.txt", "val_glint_ag_corr.txt"),
    ("ag_corr", "No-Glint", "train_noglint_ag_corr.txt", "val_noglint_ag_corr.txt"),
    ("ag_nocorr", "Glint", "train_glint_ag_nocorr.txt", "val_glint_ag_nocorr.txt"),
    ("ag_nocorr", "No-Glint", "train_noglint_ag_nocorr.txt", "val_noglint_ag_nocorr.txt"),
    ("rf", "Glint", "train_glint_rf.txt", "val_glint_rf.txt"),
    ("rf", "No-Glint", "train_noglint_rf.txt", "val_noglint_rf.txt"),
    ("bg_corr", "Glint", "train_glint_bg_corr.txt", "val_glint_bg_corr.txt"),
    ("bg_corr", "No-Glint", "train_noglint_bg_corr.txt", "val_noglint_bg_corr.txt"),
    ("bg_nocorr", "Glint", "train_glint_bg_nocorr.txt", "val_glint_bg_nocorr.txt"),
    ("bg_nocorr", "No-Glint", "train_noglint_bg_nocorr.txt", "val_noglint_bg_nocorr.txt"),
    ("full_nocorr", "Glint", "train_glint_full_nocorr.txt", "val_glint_full_nocorr.txt"),
    ("full_nocorr", "No-Glint", "train_noglint_full_nocorr.txt", "val_noglint_full_nocorr.txt"),
]

def parse_metrics_file(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: File not found -> {filepath}")
        return [(None, None, None)] * 5

    with open(filepath, 'r') as f:
        content = f.read()

    intervals_data = []

    # 1. Parse Overall
    bias_m = re.search(r"Bias:\s*([\d\.-]+|nan)", content)
    rmse_m = re.search(r"RMSE:\s*([\d\.-]+|nan)", content)
    mae_m  = re.search(r"MAE:\s*([\d\.-]+|nan)", content)

    def to_float(match):
        if not match or match.group(1) == "nan":
            return None
        return float(match.group(1))

    intervals_data.append((to_float(bias_m), to_float(rmse_m), to_float(mae_m)))

    # 2. Parse Binned Intervals
    bin_blocks = re.findall(r"For heights between.*?\nBias:\s*([\d\.-]+|nan)\nRMSE:\s*([\d\.-]+|nan)\nMAE:\s*([\d\.-]+|nan)", content)

    for b_bias, b_rmse, b_mae in bin_blocks:
        b_b = float(b_bias) if b_bias != "nan" else None
        b_r = float(b_rmse) if b_rmse != "nan" else None
        b_m = float(b_mae)  if b_mae  != "nan" else None
        intervals_data.append((b_b, b_r, b_m))

    return intervals_data

def format_val(val):
    if val is None:
        return "N/A"
    elif val > 0:
        return f"+{val:.3f}"
    elif val < 0:
        return f"-{abs(val):.3f}"
    else:
        return "0.000"

intervals = [
    "Overall",
    "$0$ to $-4$\\,m",
    "$-4$ to $-6$\\,m",
    "$-6$ to $-8$\\,m",
    "$-8$ to $-12$\\,m"
]

def generate_latex_table(model_list, table_num, caption_text, label_text):
    latex_table_rows = []

    for model_name in model_list:
        model_configs = [cfg for cfg in models_config if cfg[0] == model_name]
        escaped_model_name = model_name.replace("_", "\\_")

        model_lines = []
        for c_idx, (m_name, corr_label, train_file, val_file) in enumerate(model_configs):
            train_path = os.path.join(metrics_dir, f"metrics_{train_file}")
            val_path   = os.path.join(metrics_dir, f"metrics_{val_file}")

            train_data = parse_metrics_file(train_path)
            val_data   = parse_metrics_file(val_path)

            for i_idx, interval_name in enumerate(intervals):
                tr_b, tr_r, tr_m   = train_data[i_idx]
                val_b, val_r, val_m = val_data[i_idx]

                m_str = f"\\multirow{{10}}{{*}}{{\\textbf{{{escaped_model_name}}}}}" if (c_idx == 0 and i_idx == 0) else ""
                c_str = f"\\multirow{{5}}{{*}}{{{corr_label}}}" if i_idx == 0 else ""

                row = (
                    f"{m_str:<26} & {c_str:<20} & {interval_name:<20} & "
                    f"{format_val(tr_b):>7} & {format_val(tr_r):>5} & {format_val(tr_m):>5} & & "
                    f"{format_val(val_b):>7} & {format_val(val_r):>5} & {format_val(val_m):>5} \\\\"
                )
                model_lines.append(row)

            if c_idx < len(model_configs) - 1:
                model_lines.append("\\cmidrule(lr){2-10}")

        latex_table_rows.append("\n".join(model_lines))

    table_code = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{\\textbf{{Table {table_num}:}} {caption_text}}}
\\label{{{label_text}}}
\\small
\\setlength{{\\tabcolsep}}{{3.5pt}}
\\renewcommand{{\\arraystretch}}{{1.15}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{lll rrr c rrr}}
\\toprule
& & & \\multicolumn{{3}}{{c}}{{\\textbf{{Training Set (\\textit{{MainAOI}})}}}} & & \\multicolumn{{3}}{{c}}{{\\textbf{{Validation Set (\\textit{{SouthernRecon}})}}}} \\\\
\\cmidrule(lr){{4-6}} \\cmidrule(lr){{8-10}}
\\textbf{{Model}} & \\textbf{{Correction}} & \\textbf{{Depth Interval}} & \\textbf{{Bias}} & \\textbf{{RMSE}} & \\textbf{{MAE}} & & \\textbf{{Bias}} & \\textbf{{RMSE}} & \\textbf{{MAE}} \\\\
\\midrule

""" + "\n\\midrule\n\n".join(latex_table_rows) + f"""
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}"""
    
    return table_code

# Pass ALL 7 models in the list parameter:
all_models = [
    "full_corr",
    "ag_corr",
    "ag_nocorr",
    "rf",
    "bg_corr",
    "bg_nocorr",
    "full_nocorr"
]

table = generate_latex_table(
    all_models,
    1,
    "Detailed error metrics (Bias, RMSE, MAE) across depth intervals for all SDB models (15 August 2023). Values are reported in meters ($m$) for Training (\\textit{MainAOI}) and Validation (\\textit{SouthernRecon}) datasets. Validation depth maxes out at $-7.71\\,\\text{m}$, making the $[-8, -12\\,\\text{m}]$ interval not applicable (N/A).",
    "tab:sdb_model_overview_all"
)

output_file = os.path.join(metrics_dir, "latex_table_output.tex")
os.makedirs(metrics_dir, exist_ok=True)
with open(output_file, "w") as f:
    f.write(table)

print(f"Successfully generated full LaTeX table containing all 7 models! Saved to: {output_file}")