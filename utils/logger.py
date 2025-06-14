import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler

import torch.distributed as dist
from colorama import Fore, Style


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": Fore.BLUE,
        "INFO": Fore.WHITE,
        "SUCCESS": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        if not record.exc_info:
            level = record.levelname
            if level in self.COLORS:
                record.msg = (
                    f"{self.COLORS[level]}{record.msg}{Style.RESET_ALL}"
                )
        return super().format(record)


class CustomLogger(logging.Logger):
    def __init__(self, name, output_path, log_prefix="log.log", rank=0):
        super().__init__(name)
        self.rank = rank
        self.setLevel(logging.DEBUG)

        # Add custom SUCCESS level
        logging.SUCCESS = 25
        logging.addLevelName(logging.SUCCESS, "SUCCESS")

        # Set up file handler (only for main process)
        if rank == 0:
            log_file = os.path.join(output_path, log_prefix)
            os.makedirs(output_path, exist_ok=True)
            self.file_handler = RotatingFileHandler(
                log_file, maxBytes=10 * 1024 * 1024, backupCount=5
            )
            self.file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )
            self.file_handler.setFormatter(file_formatter)
            self.addHandler(self.file_handler)

        self.console_handler = logging.StreamHandler(sys.stdout)
        self.console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter("%(message)s")
        self.console_handler.setFormatter(console_formatter)
        self.addHandler(self.console_handler)

        # Store original stdout and stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        # Redirect stdout and stderr
        sys.stdout = self.StdoutRedirector(self)
        sys.stderr = self.StderrRedirector(self)

    def _log(self, level, msg, args, **kwargs):
        if self.rank == 0 or level >= logging.WARNING:
            super()._log(level, msg, args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        """
        Detailed information, typically of interest only when diagnosing problems.
        """
        self._log(logging.DEBUG, msg, args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """
        General information about program execution.
        """
        self._log(logging.INFO, msg, args, **kwargs)

    def success(self, msg, *args, **kwargs):
        """
        Indicates a successful operation or milestone.
        """
        self._log(logging.SUCCESS, msg, args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """
        An indication that something unexpected happened, or indicative of some problem in the near future.
        """
        self._log(logging.WARNING, msg, args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """
        Due to a more serious problem, the software has not been able to perform some function.
        """
        self._log(logging.ERROR, msg, args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """
        A serious error, indicating that the program itself may be unable to continue running.
        """
        self._log(logging.CRITICAL, msg, args, **kwargs)

    def section(self, title):
        """
        Logs a section header to visually separate different parts of the log output.
        This creates a prominent, centered title surrounded by '=' characters for easy visual identification.
        Please always use this function to separate different running stages.
        """
        self.info(f"\n{'=' * 80}\n{title:^80}\n{'=' * 80}\n")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original stdout and stderr
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        for handler in self.handlers:
            handler.close()

    class StdoutRedirector:
        def __init__(self, logger):
            self.logger = logger
            self.original_stdout = logger.original_stdout

        def write(self, msg):
            if msg.strip():  # Avoid empty messages
                self.logger.info(msg.rstrip())

        def flush(self):
            self.original_stdout.flush()

        def isatty(self):
            return self.original_stdout.isatty()

        def __getattr__(self, name):
            return getattr(self.original_stdout, name)

    class StderrRedirector:
        def __init__(self, logger):
            self.logger = logger
            self.original_stderr = logger.original_stderr

        def write(self, msg):
            if msg.strip():  # Avoid empty messages
                self.logger.error(msg.rstrip())

        def flush(self):
            self.original_stderr.flush()

        def isatty(self):
            return self.original_stderr.isatty()

        def __getattr__(self, name):
            return getattr(self.original_stderr, name)


class _GlobalLogger:
    _instance = None

    def __init__(self):
        self.logger = None
        self.rank = None

    def init(self, output_path, log_prefix="log.log"):
        # Determine rank
        if dist.is_initialized():
            self.rank = dist.get_rank()
        else:
            # If running without torchrun, set rank to 0
            self.rank = 0

        if self.logger is None:
            self.logger = self.setup_logger(output_path, log_prefix)

    def setup_logger(self, output_path, log_prefix="event.log"):
        if output_path.endswith(".txt") or output_path.endswith(".log"):
            log_file = output_path
            output_dir = os.path.dirname(output_path)
        else:
            output_dir = output_path
            log_file = os.path.join(output_dir, log_prefix)

        os.makedirs(output_dir, exist_ok=True)

        base, ext = os.path.splitext(log_file)
        log_file = f"{base}-{time.strftime('%Y-%m-%d-%H-%M-%S')}{ext}"

        logger = CustomLogger(
            "main_logger", output_dir, os.path.basename(log_file), self.rank
        )
        return logger

    def __getattr__(self, name):
        if self.logger is None:
            raise RuntimeError(
                "Logger not initialized. Call logger.init() first."
            )
        return getattr(self.logger, name)


logger = _GlobalLogger()


if __name__ == "__main__":
    logger.init("./output", 0)
    logger.info("This is an info message")
    logger.success("This is a success message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
