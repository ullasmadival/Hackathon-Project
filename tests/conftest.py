import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "code" / "business_entity_resolution" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
