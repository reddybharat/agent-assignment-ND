import subprocess
import sys


def main():
    """Run tests with pytest."""
    print("Running AI Assistant Tests")
    print("=" * 30)
    
    # Run pytest with basic options
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v"]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\nAll tests passed!")
        return 0
    except subprocess.CalledProcessError:
        print("\nSome tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
