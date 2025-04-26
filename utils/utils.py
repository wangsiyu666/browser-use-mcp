import time
from typing import (
    Dict,
    Optional
)
import os
from pathlib import Path



def get_latest_files(directory: str, files_types: list = [".webm", ".zip"]) -> Dict[str, Optional[str]]:
    latest_files: Dict[str, Optional[str]] = {ext: None for ext in files_types}

    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        return latest_files

    for files_type in files_types:
        try:
            matches = list(Path(directory).rglob(f"*{files_type}"))
            if matches:
                latest = max(matches, key=lambda p: p.stat().st_mtime)
                if time.time() - latest.stat().st_mtime > 1.0:
                    latest_files[files_type] = str(latest)
        except Exception as e:
            print(f"Error getting latest {files_type} file: {e}")
    return latest_files


