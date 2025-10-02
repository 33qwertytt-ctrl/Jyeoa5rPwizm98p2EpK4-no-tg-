import logging
import os


def setup_logger():
    # Получаем логгер
    logger = logging.getLogger('azzraelcode-yt')

    # Проверяем, не настроен ли логгер ранее
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Формат логов
    formatter = logging.Formatter('%(asctime)s %(levelname)s | %(message)s', datefmt='%m/%d %H:%M:%S')

    # Консольный обработчик
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Файловый обработчик
    os.makedirs('logs', exist_ok=True)
    file_handler = logging.FileHandler('logs/azzraelcode.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger