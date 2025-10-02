import logging
import os
import pandas as pd
import numpy as np
import ta.trend
import ta.volatility
from dotenv import load_dotenv
import glob
import traceback

load_dotenv('.env')

logger = logging.getLogger('azzraelcode-yt')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s | %(message)s')

log_dir = r'C:\Users\Егор\Downloads\azzyt-okx-main\backtest_logs'
os.makedirs(log_dir, exist_ok=True)

log_files = glob.glob(os.path.join(log_dir, 'backtest_*.log'))
if log_files:
    max_num = max(int(os.path.basename(f).split('_')[1].split('.log')[0]) for f in log_files)
    log_file = os.path.join(log_dir, f'backtest_{max_num + 1}.log')
    txt_file = os.path.join(log_dir, f'backtest_{max_num + 1}.txt')
else:
    log_file = os.path.join(log_dir, 'backtest_1.log')
    txt_file = os.path.join(log_dir, 'backtest_1.txt')

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s | %(message)s'))
logger.addHandler(file_handler)

class Backtest:
    def __init__(self, use_sl=True):
        self.candles_file = 'data/candles_data.csv'
        self.macd_fast = int(os.getenv('MACD_FAST_BACKTEST', os.getenv('MACD_FAST', 12)))
        self.macd_slow = int(os.getenv('MACD_SLOW_BACKTEST', os.getenv('MACD_SLOW', 26)))
        self.macd_signal = int(os.getenv('MACD_SIGNAL_BACKTEST', os.getenv('MACD_SIGNAL', 9)))
        self.bb_window = int(os.getenv('BB_WINDOW_BACKTEST', os.getenv('BB_WINDOW', 20)))
        self.bb_std = float(os.getenv('BB_STD_BACKTEST', os.getenv('BB_STD', 2.0)))
        self.bb_min_width = float(os.getenv('BB_MIN_WIDTH_BACKTEST', os.getenv('BB_MIN_WIDTH', 0.0003)))
        self.stop_loss_pct = float(os.getenv('STOP_LOSS_PCT_BACKTEST', os.getenv('STOP_LOSS_PCT', 0.005)))
        self.initial_risk_pct = float(os.getenv('INITIAL_RISK_PCT_BACKTEST', os.getenv('INITIAL_RISK_PCT', 0.05)))
        self.leverage = int(os.getenv('LEVERAGE_BACKTEST', os.getenv('LEVERAGE', 10)))
        self.lot_size = 0.001
        self.initial_balance = 1000.0
        self.balance = self.initial_balance
        self.position = {'side': None, 'size': 0, 'entry_price': 0}
        self.current_trend = None
        self.trades = []
        self.use_sl = use_sl
        logger.info(f"Бэктест запущен с SL={use_sl}, баланс старт={self.initial_balance} USDT")
        logger.info(f"Параметры: MACD({self.macd_fast},{self.macd_slow},{self.macd_signal}), BB({self.bb_window},{self.bb_std},{self.bb_min_width}), SL_PCT={self.stop_loss_pct}, RISK_PCT={self.initial_risk_pct}, LEV={self.leverage}")

    def calculate_sz(self, price):
        if price <= 0:
            return 0
        risk_amount = self.balance * self.initial_risk_pct
        sz = (risk_amount / price) * self.leverage
        sz = max(round(sz, 3), self.lot_size)
        return sz

    def close_position(self, price, timestamp, sl_triggered=False):
        if self.position['side'] is None or self.position['size'] == 0:
            return
        entry_price = self.position['entry_price']
        size = self.position['size']
        if self.position['side'] == 'long':
            profit = (price - entry_price) * size * self.leverage
        else:
            profit = (entry_price - price) * size * self.leverage
        self.balance += profit
        self.trades.append({
            'timestamp': timestamp,
            'side': self.position['side'],
            'type': 'close',
            'size': size,
            'entry_price': entry_price,
            'exit_price': price,
            'profit': profit,
            'sl_triggered': sl_triggered
        })
        self.position = {'side': None, 'size': 0, 'entry_price': 0}

    def check_stop_loss(self, price):
        if self.position['side'] == 'long' and price <= self.position['entry_price'] * (1 - self.stop_loss_pct):
            return True
        if self.position['side'] == 'short' and price >= self.position['entry_price'] * (1 + self.stop_loss_pct):
            return True
        return False

    def run(self):
        try:
            if not os.path.exists(self.candles_file):  # Новое: проверка существования файла
                logger.error(f"Файл {self.candles_file} не найден. Бэктест пропущен.")
                return
            df = pd.read_csv(self.candles_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp', 'close'])
            df = df.sort_values('timestamp')
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df = df.ffill().bfill()
            df = df.dropna()
            logger.info(f"Данные загружены: {len(df)} строк")
            macd = ta.trend.MACD(df['close'], window_fast=self.macd_fast, window_slow=self.macd_slow, window_sign=self.macd_signal)
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()
            bb = ta.volatility.BollingerBands(df['close'], window=self.bb_window, window_dev=self.bb_std)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
            prev_macd_hist = df['macd_histogram'].shift(1)
            cross_up = (df['macd_histogram'] > 0) & (prev_macd_hist < 0)
            cross_down = (df['macd_histogram'] < 0) & (prev_macd_hist > 0)
            for idx, row in df.iterrows():
                timestamp = row['timestamp']
                current_price = row['close']
                bb_width = row['bb_width']
                if bb_width < self.bb_min_width:
                    continue
                if self.use_sl and self.position['side'] is not None and self.check_stop_loss(current_price):
                    self.close_position(current_price, timestamp, sl_triggered=True)
                if cross_up[idx]:
                    if self.position['side'] == 'short':
                        self.close_position(current_price, timestamp)
                    if self.position['side'] != 'long':
                        sz = self.calculate_sz(current_price)
                        if sz >= self.lot_size:
                            self.position = {'side': 'long', 'size': sz, 'entry_price': current_price}
                            self.trades.append({
                                'timestamp': timestamp,
                                'side': 'long',
                                'type': 'open',
                                'size': sz,
                                'entry_price': current_price,
                                'exit_price': None,
                                'profit': 0,
                                'sl_triggered': False
                            })
                elif cross_down[idx]:
                    if self.position['side'] == 'long':
                        self.close_position(current_price, timestamp)
                    if self.position['side'] != 'short':
                        sz = self.calculate_sz(current_price)
                        if sz >= self.lot_size:
                            self.position = {'side': 'short', 'size': sz, 'entry_price': current_price}
                            self.trades.append({
                                'timestamp': timestamp,
                                'side': 'short',
                                'type': 'open',
                                'size': sz,
                                'entry_price': current_price,
                                'exit_price': None,
                                'profit': 0,
                                'sl_triggered': False
                            })
            if self.position['side'] is not None and self.position['size'] > 0:
                current_price = df['close'].iloc[-1]
                self.close_position(current_price, df['timestamp'].iloc[-1], sl_triggered=False)
            if self.trades:
                trades_df = pd.DataFrame(self.trades)
                trades_df.to_csv('data/backtest_results.csv', index=False)
                total_profit = self.balance - self.initial_balance
                profit_pct = (total_profit / self.initial_balance) * 100
                closes = [t for t in self.trades if t['type'] == 'close']
                wins = len([t for t in closes if t['profit'] > 0])
                win_rate = (wins / len(closes)) * 100 if len(closes) > 0 else 0
                num_trades = len(closes)
                sl_trades = len([t for t in closes if t['sl_triggered']])
                metrics = (
                    f"Бэктест завершён (SL={self.use_sl}):\n"
                    f"Итоговая прибыль: {total_profit:.2f} USDT ({profit_pct:.2f}%)\n"
                    f"Win-rate: {win_rate:.2f}% ({wins}/{num_trades} сделок)\n"
                    f"Количество сделок: {num_trades}, из них по SL: {sl_trades}"
                )
                logger.info(metrics)
                print(metrics)
                with open(txt_file, 'a' if self.use_sl else 'w') as f:
                    f.write(metrics + '\n\n')
            else:
                logger.warning("Нет сделок в бэктесте — проверь сигналы cross или BB width")
        except Exception as e:
            logger.error(f"Ошибка в бэктесте: {traceback.format_exc()}")

if __name__ == '__main__':
    print("Бэктест без SL:")
    backtest_no_sl = Backtest(use_sl=False)
    backtest_no_sl.run()
    print("\nБэктест с SL:")
    backtest_sl = Backtest(use_sl=True)
    backtest_sl.run()
    logger.removeHandler(file_handler)
    file_handler.close()