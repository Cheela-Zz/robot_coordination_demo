import hashlib
import json
from pathlib import Path
import time


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    # Antivirus/indexing can briefly hold a Windows destination open.
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.05 * 2 ** attempt)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
