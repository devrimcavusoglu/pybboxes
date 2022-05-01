from tests.utils import shell, validate_and_exit

if __name__ == "__main__":
    sts_tests = shell("pytest --cov pybboxes --cov-report term-missing --cov-report xml")
    validate_and_exit(tests=sts_tests)
