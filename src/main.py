import numpy as np

from models.bathemetry_utils import run_fit_pipeline


# Model Functions
def lin_simple(x, a, b):
    return a * x + b

def lin_full(x, a1, a2, a3, a4, b):
    return a1 * x[0] + a2 * x[1] + a3 * x[2] + a4 * x[3] + b

def lin_simple_corr(x, a, b, c1, c2):
    return a * (np.log10(10000 * (x[0] + c1 * x[2])) / np.log10(10000 * (x[1] + c2 * x[2]))) + b

def lin_full_corr(x, a1, a2, a3, a4, b, c1, c2, c3, c4):
    ag = np.log10(10000 * (x[0] + c1 * x[4])) / np.log10(10000 * (x[2] + c3 * x[4]))
    bg = np.log10(10000 * (x[1] + c2 * x[4])) / np.log10(10000 * (x[2] + c3 * x[4]))
    ar = np.log10(10000 * (x[0] + c1 * x[4])) / np.log10(10000 * (x[3] + c4 * x[4]))
    br = np.log10(10000 * (x[1] + c2 * x[4])) / np.log10(10000 * (x[3] + c4 * x[4]))
    return a1 * ag + a2 * bg + a3 * ar + a4 * br + b

# --- Execution Entry Points ---

def run_simple_fit():
    run_fit_pipeline(
        band_paths={'ratio': 'data/processed/aero_over_green.tif'},
        fit_func=lin_simple,
        p0=[0, 0],
        apply_outlier_filter=True,
        train_name="train_noglint_ag_nocorr",
        val_name="val_noglint_ag_nocorr"
    )

def run_full_fit():
    run_fit_pipeline(
        band_paths={
            'bg': 'data/processed/blue_over_green.tif',
            'br': 'data/processed/blue_over_red.tif',
            'ag': 'data/processed/aero_over_green.tif',
            'ar': 'data/processed/aero_over_red.tif',
            'nir': 'data/processed/nir_smooth.tif'
        },
        fit_func=lin_full,
        p0=[0, 0, 0, 0, 0],
        train_name="train_glint_full_nocorr",
        val_name="val_glint_full_nocorr"
    )

def run_simple_fit_corr():
    run_fit_pipeline(
        band_paths={
            'i': 'data/processed/blue.tif',
            'j': 'data/processed/green.tif',
            'nir': 'data/processed/nir_smooth.tif'
        },
        fit_func=lin_simple_corr,
        p0=[-40, 0, 0.1, 0.1],
        bounds=((-np.inf, -np.inf, 0, 0), (np.inf, np.inf, 10, 10)),
        train_name="train_noglint_bg_corr",
        val_name="val_noglint_bg_corr"
    )

def run_full_fit_corr():
    run_fit_pipeline(
        band_paths={
            'a': 'data/processed/aero.tif',
            'b': 'data/processed/blue.tif',
            'g': 'data/processed/green.tif',
            'r': 'data/processed/red.tif',
            'nir': 'data/processed/nir_smooth.tif'
        },
        fit_func=lin_full_corr,
        p0=[0, 0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1],
        bounds=(
            (-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0, 0, 0, 0),
            (np.inf, np.inf, np.inf, np.inf, np.inf, 10, 10, 10, 10)
        ),
        train_name="train_glint_full_corr",
        val_name="val_glint_full_corr"
    )

if __name__ == "__main__":
    run_simple_fit()
    run_full_fit()
    run_simple_fit_corr()
    run_full_fit_corr()