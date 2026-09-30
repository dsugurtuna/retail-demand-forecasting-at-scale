"""
Structured logging configuration.

Provides production-grade logging with:
- JSON formatting for production
- Console formatting for development
- Log rotation
- Context injection
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    """JSON log formatter for production environments."""

    def __init__(
        self,
        include_timestamp: bool = True,
        include_level: bool = True,
        include_name: bool = True,
        extra_fields: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self.include_name = include_name
        self.extra_fields = extra_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {}

        if self.include_timestamp:
            log_data["timestamp"] = datetime.utcnow().isoformat() + "Z"

        if self.include_level:
            log_data["level"] = record.levelname

        if self.include_name:
            log_data["logger"] = record.name

        log_data["message"] = record.getMessage()

        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "exc_info",
                "exc_text",
                "thread",
                "threadName",
                "message",
            }:
                log_data[key] = value

        # Add configured extra fields
        log_data.update(self.extra_fields)

        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for development."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[41m",  # Red background
    }
    RESET = "\033[0m"

    def __init__(self, fmt: str | None = None, datefmt: str | None = None):
        super().__init__(fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")

        # Save original levelname
        original_levelname = record.levelname

        # Apply color
        record.levelname = f"{color}{record.levelname}{self.RESET}"

        # Format the record
        result = super().format(record)

        # Restore original levelname
        record.levelname = original_levelname

        return result


def setup_logging(
    level: str | int = "INFO",
    format_type: str = "console",
    log_file: str | Path | None = None,
    include_timestamp: bool = True,
    extra_fields: dict[str, Any] | None = None,
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: "console" (colored) or "json"
        log_file: Optional file path for file logging
        include_timestamp: Include timestamp in logs
        extra_fields: Extra fields to include in JSON logs
    """
    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if format_type == "json":
        formatter = JsonFormatter(include_timestamp=include_timestamp, extra_fields=extra_fields)
    else:
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        formatter = ColoredFormatter(fmt, datefmt)

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)

        # Always use JSON for file logging
        file_handler.setFormatter(JsonFormatter(include_timestamp=True, extra_fields=extra_fields))

        root_logger.addHandler(file_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("s3transfer").setLevel(logging.WARNING)
    logging.getLogger("lightgbm").setLevel(logging.WARNING)


def get_logger(
    name: str, level: str | int | None = None, extra: dict[str, Any] | None = None
) -> logging.Logger:
    """
    Get a logger with optional configuration.

    Args:
        name: Logger name (usually __name__)
        level: Optional specific level for this logger
        extra: Extra context to include in all messages

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)

    if level is not None:
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        logger.setLevel(level)

    if extra:
        logger = logging.LoggerAdapter(logger, extra)

    return logger


class LogContext:
    """
    Context manager for adding temporary context to logs.

    Example:
        >>> with LogContext(request_id="abc123"):
        ...     logger.info("Processing request")
    """

    _context: dict[str, Any] = {}

    def __init__(self, **kwargs):
        self.new_context = kwargs
        self.previous_context = {}

    def __enter__(self):
        # Save previous values
        for key in self.new_context:
            if key in LogContext._context:
                self.previous_context[key] = LogContext._context[key]

        # Update context
        LogContext._context.update(self.new_context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous values
        for key in self.new_context:
            if key in self.previous_context:
                LogContext._context[key] = self.previous_context[key]
            else:
                LogContext._context.pop(key, None)
        return False

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        return cls._context.get(key, default)

    @classmethod
    def as_dict(cls) -> dict[str, Any]:
        return cls._context.copy()
