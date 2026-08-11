import argparse
import logging

from src.config import Settings
from src.pipeline import run_ingestion


def main():
    parser = argparse.ArgumentParser(description="Construit l'index FAISS du RAG.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reconstruit l'index même si le corpus n'a pas changé.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    result = run_ingestion(Settings.from_environment(), force=args.force)
    logging.getLogger(__name__).info("Résultat du pipeline : %s", result.status)


if __name__ == "__main__":
    main()
