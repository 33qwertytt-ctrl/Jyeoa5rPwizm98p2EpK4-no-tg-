import logging
from bot.Logger import setup_logger
from bot.Bot import Bot
from dotenv import load_dotenv
import multiprocessing
import pandas as pd
import os
import time
import pattern_finder

load_dotenv()

logger = setup_logger()

def run_predictor():
    pattern_finder.main()

if __name__ == '__main__':
    try:
        os.makedirs('data', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        predictor_process = multiprocessing.Process(target=run_predictor)
        predictor_process.start()
        Bot().run()
    except KeyboardInterrupt:
        logger.info("Бот остановлен вручную!")
    except Exception as e:
        logger.error(f"Общая ошибка: {str(e)}")
    finally:
        predictor_process.terminate()
        predictor_process.join()
        logger.info("Все процессы завершены")