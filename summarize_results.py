from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cs7643_pneumonia.cli import summarize_results_main


if __name__ == "__main__":
    summarize_results_main(sys.argv[1:])
