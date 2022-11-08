import os
import logging
import logging_loki
from rich.logging import RichHandler

logger = logging.getLogger("topvecsim")
logger.propagate = False

# Add console handler.
rich_handler = RichHandler(
    show_path=True, show_level=False, omit_repeated_times=True, rich_tracebacks=True
)
logger.addHandler(rich_handler)

if os.getenv("LOKI_URL"):
    loki_handler = logging_loki.LokiHandler(
        url=os.environ["LOKI_URL"],
        version="1",
    )

    logger.addHandler(loki_handler)

logger.setLevel(logging.INFO)
