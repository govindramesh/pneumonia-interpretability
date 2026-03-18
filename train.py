from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cs7643_pneumonia.cli import train_main


if __name__ == "__main__":
    train_main(sys.argv[1:])
