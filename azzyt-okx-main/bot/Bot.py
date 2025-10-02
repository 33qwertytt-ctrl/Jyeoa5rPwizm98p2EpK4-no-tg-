import logging
import os
import time
import glob
import traceback
import warnings
from dotenv import load_dotenv
import ccxt
import pandas as pd
import numpy as np
import ta.trend
import ta.volatility
import ta.momentum

load_dotenv('.env')

warnings.filterwarnings("ignore", category=RuntimeWarning, module="ta.trend")

logger = logging.getLogger('azzraelcode-yt')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s | %(message)s')

log_dir = r'C:\Users\Егор\Downloads\azzyt-okx-main\bot_logs'
os.makedirs(log_dir, exist_ok=True)

log_files = glob.glob(os.path.join(log_dir, 'bot_*.log'))
if log_files:
    max_num = max(int(os.path.basename(f).split('_')[1].split('.log')[0]) for f in log_files)
    log_file = os.path.join(log_dir, f'bot_{max_num + 1}.log')
else:
    log_file = os.path.join(log_dir, 'bot_1.log')

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s | %(message)s'))
logger.addHandler(file_handler)

class Bot:
    def __init__(self):
        self.exchange = ccxt.okx({
            'apiKey': os.getenv('API_KEY'),
            'secret': os.getenv('SECRET'),
            'password': os.getenv('PASSPHRASE', ''),
            'enableRateLimit': True,
        })
        self.is_demo = int(os.getenv('IS_DEMO', 0))
        if self.is_demo:
            self.exchange.set_sandbox_mode(True)
        self.symbol = os.getenv('SYMBOL', 'BTC/USDT:USDT')
        self.timeframe = os.getenv('TIMEFRAME', '1m')
        self.macd_fast = int(os.getenv('MACD_FAST', 12))
        self.macd_slow = int(os.getenv('MACD_SLOW', 26))
        self.macd_signal = int(os.getenv('MACD_SIGNAL', 9))
        self.bb_window = int(os.getenv('BB_WINDOW', 20))
        self.bb_std = float(os.getenv('BB_STD', 2.0))
        self.bb_min_width = float(os.getenv('BB_MIN_WIDTH', 0.0003))
        self.stop_loss_pct = float(os.getenv('STOP_LOSS_PCT', 0.005))
        self.use_sl = os.getenv('USE_STOP_LOSS', 'True').lower() == 'true'
        self.initial_risk_pct = float(os.getenv('INITIAL_RISK_PCT', 0.05))
        self.leverage = int(os.getenv('LEVERAGE', 10))
        self.lot_size = 0.001
        balance = self.fetch_balance_with_retry()
        self.balance = balance if balance else 1000.0
        self.initial_balance = self.balance
        self.position = {'side': None, 'size': 0, 'entry_price': 0}
        self.current_trend = None
        self.trades = []
        logger.info(f"Бот запущен с SL={self.use_sl}, баланс старт={self.initial_balance:.2f} USDT")
        logger.info(f"Бот запущен для {self.symbol} на таймфрейме {self.timeframe}")
        logger.info(f"Параметры: MACD({self.macd_fast},{self.macd_slow},{self.macd_signal}), BB({self.bb_window},{self.bb_std},{self.bb_min_width}), SL_PCT={self.stop_loss_pct}, RISK_PCT={self.initial_risk_pct}, LEV={self.leverage}")

    def fetch_balance_with_retry(self, retries=5, delay=10):
        for attempt in range(retries):
            try:
                balance = self.exchange.fetch_balance(params={'type': 'swap'})['USDT']['free']
                logger.info(f"Баланс загружен: {balance} USDT")
                return balance
            except Exception as e:
                logger.error(f"Ошибка баланса (попытка {attempt+1}/{retries}): {str(e)}")
                if attempt < retries - 1:
                    time.sleep(delay)
        return None

    def calculate_sz(self, price):
        if price <= 0:
            logger.warning("Цена <= 0, sz=0")
            return 0
        risk_amount = self.balance * self.initial_risk_pct
        sz = (risk_amount / price) * self.leverage
        sz = max(round(sz, 3), self.lot_size)
        logger.info(f"Расчет sz: risk_amount={risk_amount:.2f}, price={price:.2f}, sz={sz:.6f}")
        return sz

    def close_position(self, price, sl_triggered=False):
        if self.position['side'] is None or self.position['size'] == 0:
            return
        size = self.position['size']
        side = self.position['side']
        try:
            if side == 'long':
                order = self.exchange.create_market_sell_order(self.symbol, size)
                profit = (price - self.position['entry_price']) * size * self.leverage
            else:
                order = self.exchange.create_market_buy_order(self.symbol, size)
                profit = (self.position['entry_price'] - price) * size * self.leverage
            self.balance += profit
            logger.info(f"Закрыта {side}, size={size:.6f}, profit={profit:.2f}, SL={sl_triggered}")
            self.trades.append({'side': side, 'profit': profit, 'sl_triggered': sl_triggered})
            self.position = {'side': None, 'size': 0, 'entry_price': 0}
        except Exception as e:
            logger.error(f"Ошибка закрытия {side}: {str(e)}")

    def check_stop_loss(self, price):
        if self.position['side'] == 'long' and price <= self.position['entry_price'] * (1 - self.stop_loss_pct):
            return True
        if self.position['side'] == 'short' and price >= self.position['entry_price'] * (1 + self.stop_loss_pct):
            return True
        return False

    def get_ohlcv(self):
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=500)  # Увеличено
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.ffill().bfill()
            logger.info(f"OHLCV загружено: {len(df)} строк")
            macd = ta.trend.MACD(df['close'], window_fast=self.macd_fast, window_slow=self.macd_slow, window_sign=self.macd_signal)
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()
            bb = ta.volatility.BollingerBands(df['close'], window=self.bb_window, window_dev=self.bb_std)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
            rsi = ta.momentum.RSIIndicator(df['close'], window=14)
            df['rsi'] = rsi.rsi()
            atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14)
            df['atr'] = atr.average_true_range()
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            df.to_csv('data/candles_data.csv', index=False, mode='w')  # Перезапись
            return df
        except Exception as e:
            logger.error(f"Ошибка OHLCV: {str(e)}")
            return pd.DataFrame()

    def run(self):
        try:
            while True:
                df = self.get_ohlcv()
                if df.empty:
                    logger.warning("OHLCV пустой. Ожидание...")
                    time.sleep(60)
                    continue
                current_price = df['close'].iloc[-1]
                timestamp = df['timestamp'].iloc[-1]
                logger.info(f"Текущая цена: {current_price:.2f}")
                bb_width = df['bb_width'].iloc[-1]
                macd_hist = df['macd_histogram'].iloc[-1]
                prev_macd_hist = df['macd_histogram'].iloc[-2] if len(df) > 1 else 0
                cross_up = (macd_hist > 0) and (prev_macd_hist < 0)
                cross_down = (macd_hist < 0) and (prev_macd_hist > 0)
                logger.info(f"Сигналы: cross_up={cross_up}, cross_down={cross_down}, bb_width>{self.bb_min_width}={bb_width > self.bb_min_width}")
                if self.use_sl and self.position['side'] is not None and self.check_stop_loss(current_price):
                    self.close_position(current_price, sl_triggered=True)
                if bb_width < self.bb_min_width:
                    time.sleep(60)
                    continue
                if cross_up or cross_down:
                    trend = 'up' if cross_up else 'down'
                    if self.current_trend != trend:
                        if self.position['side'] is not None:
                            self.close_position(current_price)
                        sz = self.calculate_sz(current_price)
                        if sz >= self.lot_size:
                            side = 'long' if trend == 'up' else 'short'
                            order = self.exchange.create_market_buy_order(self.symbol, sz) if side == 'long' else self.exchange.create_market_sell_order(self.symbol, sz)
                            logger.info(f"Открыт {side} ордер: {order}")
                            self.position = {'side': side, 'size': sz, 'entry_price': current_price}
                            self.current_trend = trend
                            self.trades.append({
                                'timestamp': timestamp,
                                'side': side,
                                'type': 'open',
                                'size': sz,
                                'entry_price': current_price,
                                'exit_price': None,
                                'profit': 0,
                                'sl_triggered': False
                            })
                            logger.info(f"Смена тренда на {trend}, открыта {side}, size={sz:.6f}, entry_price={current_price:.2f}")
                if self.position['side'] is None:
                    sz = self.calculate_sz(current_price)
                    if sz >= self.lot_size:
                        side = 'long' if self.current_trend == 'up' else 'short'
                        try:
                            order = self.exchange.create_market_buy_order(self.symbol, sz) if side == 'long' else self.exchange.create_market_sell_order(self.symbol, sz)
                            logger.info(f"Открыт {side} ордер: {order}")
                            self.position = {'side': side, 'size': sz, 'entry_price': current_price}
                            self.trades.append({
                                'timestamp': timestamp,
                                'side': side,
                                'type': 'open',
                                'size': sz,
                                'entry_price': current_price,
                                'exit_price': None,
                                'profit': 0,
                                'sl_triggered': False
                            })
                            logger.info(f"Открыта позиция {side}, size={sz:.6f}, entry_price={current_price:.2f}")
                        except Exception as e:
                            logger.error(f"Ошибка при открытии позиции {side}: {traceback.format_exc()}")
                time.sleep(60)
        except Exception as e:
            logger.error(f"Ошибка в боте: {traceback.format_exc()}")

if __name__ == '__main__':
    bot = Bot()
    bot.run()
    logger.removeHandler(file_handler)
    file_handler.close()