import sys
from pathlib import Path

# Make the repo root importable so `import game.turns` works from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
