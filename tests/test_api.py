import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import app


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_liveness_is_public(self):
        response = self.client.get("/health/live")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "healthy"})

    @patch("api.main.validate_current_index")
    def test_readiness_checks_the_index(self, validate_index):
        validate_index.return_value = {"results": {"chunks": 20}}

        response = self.client.get("/health/ready")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ready", "vectors": 20})

    @patch("api.main.search_index")
    def test_query_only_requires_a_question(self, search):
        search.return_value = [
            {
                "text": "Un passage pertinent.",
                "source": "rapport.pdf",
                "page": 2,
            }
        ]

        response = self.client.post(
            "/query",
            json={"question": "Quels indicateurs ?"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["results"][0]["source"], "rapport.pdf")
        self.assertNotIn("score", response.json()["results"][0])
        self.assertIn("Un passage pertinent", response.json()["answer"])
        search.assert_called_once()
        self.assertEqual(search.call_args.kwargs["limit"], 3)

    @patch("api.main.search_index", return_value=[])
    def test_query_explains_when_nothing_is_relevant(self, _search):
        response = self.client.post(
            "/query",
            json={"question": "Quel est le prix du taro ?"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["results"], [])
        self.assertIn("Aucune information pertinente", response.json()["answer"])

    @patch("api.main.search_index", side_effect=RuntimeError("secret interne"))
    def test_internal_errors_are_not_exposed(self, _search):
        response = self.client.post(
            "/query",
            json={"question": "Quels indicateurs ?"},
        )

        self.assertEqual(response.status_code, 500)
        self.assertNotIn("secret interne", response.text)



if __name__ == "__main__":
    unittest.main()
