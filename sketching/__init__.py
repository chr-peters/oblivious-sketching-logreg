import logging

logging.basicConfig(
    level="INFO",
    format="%(asctime)s - PID: %(process)d - "
    "PName: %(processName)s - %(levelname)s - %(message)s",
)
