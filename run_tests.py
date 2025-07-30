#!/usr/bin/env python3
"""Test runner script for MongoDB model unit tests."""

import sys
import subprocess
import os
from pathlib import Path


def run_tests():
    """Run all MongoDB model tests with coverage reporting."""
    # Change to project root directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Add project root to Python path
    sys.path.insert(0, str(project_root))
    
    # Test command with coverage
    cmd = [
        "python", "-m", "pytest",
        "tests/db/mongo/",
        "-v",  # Verbose output
        "--cov=app.db.mongo",  # Coverage for mongo models
        "--cov-report=html",  # HTML coverage report
        "--cov-report=term-missing",  # Terminal coverage with missing lines
        "--tb=short"  # Short traceback format
    ]
    
    print("Running MongoDB model unit tests...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("📊 Coverage report generated in htmlcov/index.html")
        return 0
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 60)
        print("❌ Some tests failed!")
        return e.returncode
    except FileNotFoundError:
        print("❌ pytest not found. Please install test dependencies:")
        print("pip install -r tests/requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
