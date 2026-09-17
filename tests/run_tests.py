import sys
import unittest
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    tests_dir = project_root / "tests"

    loader = unittest.TestLoader()

    suite = loader.discover(
        start_dir=str(tests_dir),
        pattern="test_*.py",
        top_level_dir=str(project_root),
    )

    runner = unittest.TextTestRunner(
        verbosity=2,
    )

    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())