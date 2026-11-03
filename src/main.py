

from models.Full_fit_corr import run_full_fit_corr
from models.Full_fit import run_full_fit
from models.RF_prediction import run_rf
from models.Simple_fit_corr import run_simple_fit_cor
from models.Simple_fit import run_simple_fit

from concurrent.futures import ProcessPoolExecutor

def main():
    functions = [
        run_full_fit_corr,
        run_full_fit,
        run_rf,
        run_simple_fit_cor,
        run_simple_fit_cor,
    ]
    
    with ProcessPoolExecutor() as executor:
        # Submit all tasks and wait for them to complete
        futures = [executor.submit(fn) for fn in functions]
        
        # Optionally get results or raise exceptions if any occurred
        for future in futures:
            future.result()

if __name__ == "__main__":
    main()