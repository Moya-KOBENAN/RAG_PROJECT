import logging

from src.config import Settings
from src.pipeline import validate_current_index


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    validate_current_index(Settings.from_environment())


if __name__ == "__main__":
    main()
