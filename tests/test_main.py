import importlib
import sys
import unittest
from unittest.mock import patch


class MainDispatchTest(unittest.TestCase):
    def setUp(self):
        self.main_module = importlib.import_module("main")

    def test_dispatches_prepare_chunks_command(self):
        with patch.object(sys, "argv", ["main.py", "prepare-chunks", "--help"]):
            with patch.object(self.main_module.vector_store, "prepare_chunks_main", return_value=0) as command:
                self.assertEqual(self.main_module.main(), 0)
                self.assertEqual(sys.argv, ["main.py", "--help"])

        command.assert_called_once_with()

    def test_dispatches_upload_vector_store_command(self):
        with patch.object(sys, "argv", ["main.py", "upload-vector-store", "--dry-run"]):
            with patch.object(self.main_module.vector_store, "main", return_value=0) as command:
                self.assertEqual(self.main_module.main(), 0)
                self.assertEqual(sys.argv, ["main.py", "--dry-run"])

        command.assert_called_once_with()

    def test_legacy_scrape_options_still_route_to_scraper(self):
        with patch.object(sys, "argv", ["main.py", "--clean"]):
            with patch.object(self.main_module.scraper, "main", return_value=0) as command:
                self.assertEqual(self.main_module.main(), 0)

        command.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
