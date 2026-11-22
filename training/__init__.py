"""training - prepare the Tongji palm dataset, train the CNN and evaluate it.

The model itself lives in backend/model.py so the web app and training use the same code.
"""
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))  # so `import palm` and `import model` work from the repo root
