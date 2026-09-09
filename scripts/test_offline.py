"""Run the complete synthetic suite without inherited credentials or network IO."""

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    environment = {name: os.environ[name] for name in ("PATH", "TMPDIR", "LANG") if name in os.environ}
    environment.update(PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(ROOT)]))
    code = r'''
import socket
import sys
import unittest
def blocked(*args, **kwargs):
    raise AssertionError("Network access is disabled in offline tests")
socket.socket.connect = blocked
socket.socket.connect_ex = blocked
socket.create_connection = blocked
socket.getaddrinfo = blocked
suite = unittest.defaultTestLoader.discover("tests", top_level_dir=".")
result = unittest.TextTestRunner(verbosity=2 if "--verbose" in sys.argv else 1).run(suite)
raise SystemExit(not result.wasSuccessful())
'''
    return subprocess.call([sys.executable, "-c", code, *sys.argv[1:]], env=environment, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
