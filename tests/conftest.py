import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_gen import build_frames, write_sqlite


@pytest.fixture(scope="session")
def frames():
    return build_frames()


@pytest.fixture(scope="session")
def sqlite_db(tmp_path_factory, frames):
    path = tmp_path_factory.mktemp("db") / "sales.sqlite"
    write_sqlite(frames, path)
    return path
