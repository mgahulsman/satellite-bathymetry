import os
import re

# Directory containing the metrics txt files
metrics_dir = os.path.join("data", "output", "linear")

# List of all 7 models with Glint and No-Glint configurations
models_config = [
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

def parse_overall_metrics(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: File not found -> {filepath}")
        return None, None, None

    with open(filepath, 'r') as f:
        content = f.read()

    bias_m = re.search(r"Bias:\s*([\d\.-]+|nan)", content)
    rmse_m = re.search(r"RMSE:\s*([\d\.-]+|nan)", content)
    mae_m  = re.search(r"MAE:\s*([\d\.-]+|nan)", content)

    def to_float(match):
        if not match or match.group(1) == "nan":
            return None
        return float(match.group(1))

    return to_float(bias_m), to_float(rmse_m), to_float(mae_m)

def evaluate_overfitting(tr_rmse, val_rmse, tr_bias, val_bias):
    if tr_rmse is None or val_rmse is None or tr_rmse == 0:
        return "N/A", "N/A", "N/A"
    
    ratio = val_rmse / tr_rmse
    bias_shift = val_bias - tr_bias if (val_bias is not None and tr_bias is not None) else None
    abs_bias_shift = abs(bias_shift) if bias_shift is not None else None

    # Overfitting & Domain Collapse Criteria
    if ratio > 2.5 or (abs_bias_shift is not None and abs_bias_shift > 1.5):
        status = "Severe Overfit / Domain Collapse"
    elif ratio > 1.5 or (abs_bias_shift is not None and abs_bias_shift > 0.5):
        status = "Moderate Overfit"
    else:
        status = "Robust Generalization"

    ratio_str = f"{ratio:.2f}"
    shift_str = f"{bias_shift:+.3f}" if bias_shift is not None else "N/A"

    return status, ratio_str, shift_str

def generate_overfitting_table():
    rows = []
    for m_name, corr_label, train_file, val_file in models_config:
        tr_path = os.path.join(metrics_dir, f"metrics_{train_file}")
        val_path = os.path.join(metrics_dir, f"metrics_{val_file}")

        tr_b, tr_r, tr_m = parse_overall_metrics(tr_path)
        val_b, val_r, val_m = parse_overall_metrics(val_path)

        status, ratio_str, shift_str = evaluate_overfitting(tr_r, val_r, tr_b, val_b)

        escaped_mname = m_name.replace("_", "\\_")
        tr_b_str = f"{tr_b:+.3f}" if tr_b is not None else "N/A"
        tr_r_str = f"{tr_r:.3f}" if tr_r is not None else "N/A"
        val_b_str = f"{val_b:+.3f}" if val_b is not None else "N/A"
        val_r_str = f"{val_r:.3f}" if val_r is not None else "N/A"

        row = (
            f"\\textbf{{{escaped_mname}}} & {corr_label} & "
            f"{tr_b_str} & {tr_r_str} & "
            f"{val_b_str} & {val_r_str} & "
            f"{ratio_str} & {shift_str} & "
            f"\\textit{{{status}}} \\\\"
        )
        rows.append(row)

    table_latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{\\textbf{{Table: SDB Model Overfitting and Domain Generalization Analysis.}} "
         "Summary of training and out-of-domain validation metrics, displaying the RMSE Ratio "
         "($R_{{\\text{{OOD}}}} = \\text{{RMSE}}_{{\\text{{val}}}} / \\text{{RMSE}}_{{\\text{{train}}}}$) "
         "and Systematic Bias Shift ($\\Delta\\text{{Bias}} = \\text{{Bias}}_{{\\text{{val}}}} - \\text{{Bias}}_{{\\text{{train}}}}$).}}
\\label{{tab:sdb_overfitting_analysis}}
\\small
\\setlength{{\\tabcolsep}}{{4pt}}
\\renewcommand{{\\arraystretch}}{{1.15}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{ll r r r r r r l}}
\\toprule
& & \\multicolumn{{2}}{{c}}{{\\textbf{{Training Set}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{Validation Set}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{Overfitting Indicators}}}} & \\\\
\\cmidrule(lr){{3-4}} \\cmidrule(lr){{5-6}} \\cmidrule(lr){{7-8}}
\\textbf{{Model}} & \\textbf{{Correction}} & \\textbf{{Bias ($m$)}} & \\textbf{{RMSE ($m$)}} & \\textbf{{Bias ($m$)}} & \\textbf{{RMSE ($m$)}} & \\textbf{{$R_{{\\text{{OOD}}}}$}} & \\textbf{{$\\Delta$Bias ($m$)}} & \\textbf{{Classification}} \\\\
\\midrule
""" + "\n".join(rows) + f"""
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}"""

    return table_latex

if __name__ == "__main__":
    table_code = generate_overfitting_table()
    os.makedirs(metrics_dir, exist_ok=True)
    out_tex_path = os.path.join(metrics_dir, "overfitting_analysis_table.tex")
    
    with open(out_tex_path, "w") as f:
        f.write(table_code)

    print(f"Overfitting analysis table generated and saved to: {out_tex_path}")