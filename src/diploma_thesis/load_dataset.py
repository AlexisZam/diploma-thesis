"""Load a dataset from a url."""

from pathlib import Path, PurePath
from typing import Self
from urllib import parse

import requests

DIPLOMA_THESIS_CACHE = Path("~/.cache/diploma-thesis/").expanduser()
DATASETS_CACHE = DIPLOMA_THESIS_CACHE.joinpath("datasets/")

TIMEOUT = (3.05, 27)
CHUNK_SIZE = 4096


class DatasetLoader:  # pylint: disable=too-few-public-methods
    """Dataset loader."""

    def __init__(self: Self, url: str) -> None:
        """Initialize a dataset loader."""
        self.url = url

    def _download_dataset_archive(self: Self) -> None:
        archive_name = PurePath(parse.urlparse(self.url).path).name
        archive_path = DATASETS_CACHE.joinpath(archive_name)
        if not archive_path.is_file():
            with requests.get(self.url, timeout=TIMEOUT, stream=True) as response:
                response.raise_for_status()
                with archive_path.open(mode="bw") as archive:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        archive.write(chunk)
