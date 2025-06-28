import rich
import logging
import datetime

from pathlib import Path
from typing import Literal
from torch.utils.tensorboard.writer import SummaryWriter


def get_text_logger(
    name: str,
    level: Literal[20] = logging.INFO,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def get_train_logger(
    log_dir: Path,
) -> SummaryWriter:
    log_dir = log_dir / datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    writer = SummaryWriter(log_dir)
    return writer
