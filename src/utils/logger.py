"""
logger.py
---------
Centralized logging configuration using Loguru.

Responsibilities:
- Configure console and file log sinks
- Expose a ready-to-import `logger` instance
"""

from loguru import logger  # noqa: F401

# Logger will be configured by InferencePipeline.setup()
# Import this module's `logger` throughout the project:
#   from src.utils.logger import logger
