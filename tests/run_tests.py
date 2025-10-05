#!/usr/bin/env python3
"""
Simple test runner for the RAG Agent project.
"""
import subprocess
import sys


def main():
    """Run tests with pytest."""
    print("🧪 Running RAG Agent Tests")
    print("=" * 40)
    
    # Run pytest with basic options
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v"]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ All tests passed!")
        return 0
    except subprocess.CalledProcessError:
        print("\n❌ Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
