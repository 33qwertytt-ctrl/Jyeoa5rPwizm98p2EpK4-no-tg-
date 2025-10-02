import os
import random
import subprocess
import re
from dotenv import load_dotenv, dotenv_values
import pandas as pd
import time
import warnings
import sys
from multiprocessing import Pool

warnings.filterwarnings("ignore", category=FutureWarning)

load_dotenv('.env')

param_ranges = {
    'MACD_FAST_BACKTEST': range(4, 21),
    'MACD_SLOW_BACKTEST': range(15, 41),
    'MACD_SIGNAL_BACKTEST': range(4, 21),
    'BB_WINDOW_BACKTEST': range(5, 41),
    'BB_STD_BACKTEST': [round(x * 0.1, 1) for x in range(10, 41)],
    'BB_MIN_WIDTH_BACKTEST': [round(x * 0.0005, 4) for x in range(1, 41)],
    'STOP_LOSS_PCT_BACKTEST': [round(x * 0.0005, 4) for x in range(1, 41)],
    'INITIAL_RISK_PCT_BACKTEST': [round(x * 0.005, 3) for x in range(1, 41)],
    'LEVERAGE_BACKTEST': range(1, 21)
}

def update_env(params):
    env_data = dotenv_values('.env')
    for key, value in params.items():
        env_data[key] = str(value)
    with open('.env', 'w') as f:
        for key, value in env_data.items():
            f.write(f"{key}={value}\n")

def run_backtest_wrapper(args):
    test_id, params = args
    update_env(params)
    return test_id, run_backtest(test_id)

def run_backtest(test_id):
    try:
        subprocess.run([sys.executable, 'backtest.py'], check=True, timeout=600)
        time.sleep(2)
    except Exception as e:
        print(f"Исключение при запуске backtest.py: {str(e)}")
        return 0.0, 0.0
    log_dir = r'C:\Users\Егор\Downloads\azzyt-okx-main\backtest_logs'
    txt_files = [f for f in os.listdir(log_dir) if f.endswith('.txt')]
    if not txt_files:
        return 0.0, 0.0
    latest_txt = max(txt_files, key=lambda f: int(f.split('_')[1].split('.txt')[0]))
    try:
        with open(os.path.join(log_dir, latest_txt), 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception:
        return 0.0, 0.0
    winrate_match = re.search(r"Win-rate: (\d+\.\d+)%", content)
    profit_match = re.search(r"прибыль: ([-]?\d+\.\d+) USDT", content)
    if not winrate_match or not profit_match:
        return 0.0, 0.0
    winrate = float(winrate_match.group(1))
    profit = float(profit_match.group(1))
    return winrate, profit

def main(num_iterations=1000):
    params_file = 'data/params.csv'
    results_file = 'data/results.csv'
    if os.path.exists(params_file):
        params_df = pd.read_csv(params_file)
    else:
        params_df = pd.DataFrame(columns=['test_id'] + list(param_ranges.keys()))
    if os.path.exists(results_file):
        results_df = pd.read_csv(results_file)
    else:
        results_df = pd.DataFrame(columns=['test_id', 'winrate', 'profit'])
    max_winrate = results_df['winrate'].max() if not results_df.empty else 0.0
    test_params = []
    start_id = len(results_df) + 1
    for test_id in range(start_id, start_id + num_iterations):
        params = {key: random.choice(values) for key, values in param_ranges.items()}
        test_params.append((test_id, params))
    with Pool(processes=4) as pool:
        results = pool.map(run_backtest_wrapper, test_params)
    for test_id, (winrate, profit) in results:
        if winrate > max_winrate:
            params = test_params[test_id - start_id][1]
            params_row = pd.DataFrame([[test_id] + list(params.values())], columns=params_df.columns)
            results_row = pd.DataFrame([[test_id, winrate, profit]], columns=results_df.columns)
            params_df = pd.concat([params_df, params_row], ignore_index=True)
            results_df = pd.concat([results_df, results_row], ignore_index=True)
            max_winrate = winrate
    params_df.to_csv(params_file, index=False)
    results_df.to_csv(results_file, index=False)
    results_df['score'] = results_df['profit'] * results_df['winrate']
    top_results = results_df.sort_values('score', ascending=False).head(10)
    print("Топ 10 комбинаций по score (profit * winrate):")
    print(top_results)

if __name__ == '__main__':
    main()