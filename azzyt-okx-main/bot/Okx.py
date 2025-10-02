import logging
import os
from time import sleep
import pandas as pd
from ccxt import okx
import requests
import time

logger = logging.getLogger('azzraelcode-yt')

class Okx:
    def __init__(self):
        self.api_key = os.getenv('API_KEY')
        self.api_secret = os.getenv('SECRET')
        self.passphrase = os.getenv('PASSPHRASE')
        self.name = os.getenv('NAME', 'DEMO (1)')
        self.is_demo = int(os.getenv('IS_DEMO', 0))
        self.symbol = os.getenv('SYMBOL', 'BTC/USDT:USDT')
        self.leverage = int(os.getenv('LEVERAGE', 10))
        self.margin_mode = os.getenv('MARGIN_MODE', 'isolated')
        self.stop_loss_pct = float(os.getenv('STOP_LOSS_PCT', 0.01))
        self.initial_risk_pct = float(os.getenv('INITIAL_RISK_PCT', 0.05))
        self.exchange = okx({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'password': self.passphrase,
            'enableRateLimit': True
        })
        if self.is_demo:
            self.exchange.set_sandbox_mode(True)
        logger.info(f"{self.name} OKX Auth loaded")
        for attempt in range(5):
            try:
                self.exchange.load_markets(params={'instType': 'SWAP'})
                if self.symbol not in self.exchange.markets:
                    logger.error(f"Символ {self.symbol} не найден в рынках OKX")
                    raise ValueError(f"Символ {self.symbol} не поддерживается")
                break
            except Exception as e:
                logger.error(f"Ошибка загрузки рынков (попытка {attempt + 1}/5): {str(e)}")
                if attempt < 4:
                    logger.info("Повторная попытка через 10 секунд...")
                    sleep(10)
                else:
                    raise
            except requests.exceptions.RequestException as e:
                logger.error(f"Сетевая ошибка при загрузке рынков: {str(e)}")
                raise

    def get_balance(self):
        try:
            balance = self.exchange.fetch_balance(params={'type': 'swap'})['USDT']['free']
            logger.info(f"Баланс USDT: {balance}")
            return balance
        except Exception as e:
            logger.error(f"Ошибка получения баланса: {str(e)}")
            return 0

    def check_permissions(self):
        try:
            inst_id = self.symbol.split(':')[0].replace('/', '-')
            self.exchange.private_post_account_set_leverage({
                'instId': inst_id,
                'lever': self.leverage,
                'mgnMode': self.margin_mode
            })
            logger.info(f"Плечо {self.leverage}x и режим {self.margin_mode} установлены для {inst_id}")
        except Exception as e:
            logger.error(f"Ошибка настройки плеча: {str(e)}")
            raise

    def get_ohlcv(self, symbol, timeframe='1m', limit=100):
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit, params={'instType': 'SWAP'})
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = df['timestamp'] // 1000
            logger.info(f"Получено {len(df)} свечей для {symbol} на таймфрейме {timeframe}")
            return df
        except Exception as e:
            logger.error(f"Ошибка получения OHLCV: {str(e)}")
            return pd.DataFrame()

    def get_position(self):
        try:
            positions = self.exchange.fetch_positions([self.symbol], params={'instType': 'SWAP'})
            for pos in positions:
                if pos['symbol'] == self.symbol:
                    return {
                        'side': pos['side'],
                        'size': float(pos['contracts'] or 0),
                        'avgPx': float(pos['entryPrice'] or 0)
                    }
            return {'side': None, 'size': 0, 'avgPx': 0}
        except Exception as e:
            logger.error(f"Ошибка получения позиции: {str(e)}")
            return {'side': None, 'size': 0, 'avgPx': 0}

    def get_lot_size(self):
        try:
            market = self.exchange.market(self.symbol)
            lot_size = float(market['info']['lotSz'])
            logger.info(f"Шаг лота для {self.symbol}: {lot_size}")
            return lot_size
        except Exception as e:
            logger.error(f"Ошибка получения шага лота: {str(e)}. Используется значение по умолчанию 0.001")
            return 0.001

    def round_to_lot_size(self, size, lot_size):
        if lot_size <= 0:
            logger.error("Шаг лота равен 0 или отрицательный, невозможно округлить")
            return 0
        rounded_size = round(size / lot_size) * lot_size
        return rounded_size

    def calculate_sz(self, price):
        balance = self.get_balance()
        risk_amount = balance * self.initial_risk_pct
        size = risk_amount / price
        size *= self.leverage
        lot_size = self.get_lot_size()
        rounded_size = self.round_to_lot_size(size, lot_size)
        logger.info(f"Рассчитанный размер ордера: {size:.8f}, округленный до шага лота ({lot_size}): {rounded_size:.8f}")
        return rounded_size

    def place_order(self, action, side, size, price):
        max_retries = 3
        retry_delay = 5
        for attempt in range(max_retries):
            try:
                inst_id = self.symbol.split(':')[0].replace('/', '-') + '-SWAP'
                cl_ord_id = f"ord{str(int(time.time() * 1000))[-8:]}"
                logger.info(f"Сформированный clOrdId: {cl_ord_id} (длина: {len(cl_ord_id)})")
                if size <= 0:
                    logger.error("Размер ордера равен 0 или отрицательный, ордер не отправлен")
                    return None
                current_position = self.get_position()
                logger.info(f"Текущая позиция: {current_position}")
                order_params = {
                    'instId': inst_id,
                    'tdMode': self.margin_mode,
                    'clOrdId': cl_ord_id,
                    'instType': 'SWAP'
                }
                pos_side = 'long' if side == 'long' else 'short'
                if action == 'close':
                    ord_type = 'market'
                    if current_position['side'] == side and current_position['size'] > 0:
                        order_params.update({
                            'side': 'sell' if side == 'long' else 'buy',
                            'posSide': pos_side,
                            'ordType': ord_type,
                            'sz': str(size)
                        })
                    else:
                        order_params.update({
                            'side': 'sell' if side == 'long' else 'buy',
                            'ordType': ord_type,
                            'sz': str(size)
                        })
                else:
                    ord_type = 'limit'
                    order_side = 'buy' if side == 'long' else 'sell'
                    if current_position['side'] == side and current_position['size'] > 0:
                        order_params.update({
                            'side': order_side,
                            'posSide': pos_side,
                            'ordType': ord_type,
                            'px': str(price),
                            'sz': str(size)
                        })
                    else:
                        order_params.update({
                            'side': order_side,
                            'ordType': ord_type,
                            'px': str(price),
                            'sz': str(size)
                        })
                logger.info(f"Попытка {attempt + 1}/{max_retries} отправки ордера: {order_params}")
                response = self.exchange.private_post_trade_order(order_params)
                logger.info(f"Ответ API на ордер: {response}")
                return response
            except Exception as e:
                logger.error(f"Ошибка размещения ордера (попытка {attempt + 1}/{max_retries}): {str(e)}")
                if 'code' in str(e) and '50001' in str(e):
                    if attempt < max_retries - 1:
                        logger.info(f"Сервер временно недоступен, повтор через {retry_delay} секунд...")
                        sleep(retry_delay)
                        continue
                return None
        logger.error(f"Не удалось разместить ордер после {max_retries} попыток")
        return None