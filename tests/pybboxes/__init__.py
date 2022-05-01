from pathlib import Path

TESTS_PACKAGE_DIR = Path(__file__).parent
TESTS_DIR = TESTS_PACKAGE_DIR.parent
TEST_DATA_DIR = TESTS_DIR / "test_data"
EXPECTED_OUTPUTS = TEST_DATA_DIR / "expected_outputs"
