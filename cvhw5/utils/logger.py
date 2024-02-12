import logging
import sys


def set_up_logger() -> None:
    logging.basicConfig(
        stream=sys.stdout,
        format='%(asctime)s %(module)-15s — %(funcName)-20s — %(levelname)-8s — %(message)s', level=logging.INFO,
        datefmt='%H:%M:%S')


def set_up_logging_to_file(filepath: str) -> None:
    logFormatter = logging.Formatter(
        '%(asctime)s %(module)-15s — %(funcName)-20s — %(levelname)-8s — %(message)s',
        datefmt='%H:%M:%S')

    fileHandler = logging.FileHandler(filepath)
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(logFormatter)

    logging.getLogger().addHandler(fileHandler)
