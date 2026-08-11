import unittest

from src.price_extractor import (
    extract_answer,
    extract_price_answer,
    extract_variation_answer,
)


class PriceExtractorTests(unittest.TestCase):
    def test_extracts_current_prices_from_two_regions(self):
        results = [
            {
                "text": (
                    "BULLETIN - DR YAMOUSSOUKRO / MAI 2026\n"
                    "4 Gombo frais 1 Kg 1 395 960 -31,2%"
                )
            },
            {
                "text": (
                    "BULLETIN - DR AGBOVILLE / MAI 2026\n"
                    "4 Gombo frais 1 Kg 1 245 1 270 2%"
                )
            },
        ]

        answer = extract_price_answer("Quel est le prix du gombo ?", results)

        self.assertIn("960 FCFA", answer)
        self.assertIn("1 270 FCFA", answer)
        self.assertIn("Yamoussoukro", answer)
        self.assertIn("Agboville", answer)

    def test_returns_none_without_a_matching_price_row(self):
        answer = extract_price_answer(
            "Quel est le prix du taro ?",
            [{"text": "4 Gombo frais 1 Kg 1 395 960 -31,2%"}],
        )

        self.assertIsNone(answer)

    def test_extracts_variations_from_two_regions(self):
        answer = extract_answer(
            "Quelle est la variation du prix du gombo ?",
            [
                {
                    "text": (
                        "BULLETIN - DR YAMOUSSOUKRO / MAI 2026\n"
                        "4 Gombo frais 1 Kg 1 395 960 -31,2%"
                    ),
                    "source": "bulletin.pdf",
                    "page": 1,
                },
                {
                    "text": (
                        "BULLETIN - DR AGBOVILLE / MAI 2026\n"
                        "4 Gombo frais 1 Kg 1 245 1 270 2%"
                    ),
                    "source": "bulletin-2.pdf",
                    "page": 1,
                },
            ],
        )

        self.assertIn("a diminué de 31,2% à Yamoussoukro", answer)
        self.assertIn("a augmenté de 2% à Agboville", answer)
        self.assertIn("MAI 2026", answer)

    def test_returns_none_for_variation_without_a_matching_row(self):
        answer = extract_variation_answer(
            "Quelle est la variation du prix du taro ?",
            [{"text": "4 Gombo frais 1 Kg 1 395 960 -31,2%"}],
        )

        self.assertIsNone(answer)

    def test_returns_a_general_extract_for_another_question(self):
        answer = extract_answer(
            "Donne-moi les informations sur le gombo.",
            [
                {
                    "text": "4 Gombo frais 1 Kg 1 395 960 -31,2%",
                    "source": "bulletin.pdf",
                    "page": 1,
                }
            ],
        )

        self.assertIn("Gombo frais", answer)
        self.assertIn("bulletin.pdf, page 1", answer)


if __name__ == "__main__":
    unittest.main()
