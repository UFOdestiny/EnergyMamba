import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager


def get_logger(log_dir, name, log_filename=None, level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)
    if log_filename is None:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        log_filename = f"{timestamp}.log"

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    log_path = os.path.join(log_dir, log_filename)
    fh = RotatingFileHandler(log_path, maxBytes=10 * 1024 * 1024, backupCount=50, encoding="utf-8")
    fh.setFormatter(formatter)
    fh.setLevel(level)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    ch.setLevel(level)

    logger.addHandler(fh)
    logger.addHandler(ch)

    @contextmanager
    def no_time():
        old = [h.formatter for h in logger.handlers]
        for h in logger.handlers:
            h.setFormatter(logging.Formatter("%(message)s"))
        try:
            yield
        finally:
            for h, f in zip(logger.handlers, old):
                h.setFormatter(f)

    logger.no_time = no_time

    with logger.no_time():
        logger.info("=" * 25 + "   Settings   " + "=" * 25)
    logger.info(f"Log File Path: {log_path}")
    return logger
