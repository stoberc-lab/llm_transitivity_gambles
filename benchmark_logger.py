import logging
from pathlib import Path
from datetime import datetime

def get_logger(base_dir: Path, benchmark: str, name: str):
    logs_dir = f"{base_dir}/logs/{benchmark}"
    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(f"{logs_dir}/{datetime.today().strftime('%Y%m%d')}.log")
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter('[%(asctime)s.%(msecs)03d][%(name)s][%(levelname)s] - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
