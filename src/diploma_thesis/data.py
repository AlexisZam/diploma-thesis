"""Load data."""

import dataclasses
import shutil
from pathlib import Path, PurePath
from typing import Self
from urllib import parse

import requests

DIPLOMA_THESIS_CACHE = Path("~/.cache/diploma-thesis/").expanduser()
DATA_CACHE = DIPLOMA_THESIS_CACHE.joinpath("data/")

TIMEOUT = (3.05, 27)


@dataclasses.dataclass(slots=True)
class DataLoader:
    """Data loader.

    Attributes:
        data_url: The url of the data.
    """

    data_url: str

    @property
    def _data_url_path(self: Self) -> PurePath:
        return PurePath(parse.urlparse(self.data_url).path)

    @property
    def _data_archive(self: Self) -> Path:
        return DATA_CACHE.joinpath(self._data_url_path.name)

    @property
    def _data_dir(self: Self) -> Path:
        return DATA_CACHE.joinpath(self._data_url_path.stem)

    @staticmethod
    def _create_data_cache_dir() -> None:
        DATA_CACHE.mkdir(parents=True, exist_ok=True)

    def _download_data_archive(self: Self) -> None:
        if self._data_archive.is_file():
            return
        with requests.get(self.data_url, timeout=TIMEOUT, stream=True) as response:
            response.raise_for_status()
            with self._data_archive.open(mode="bw") as data_archive:
                for chunk in response.iter_content(chunk_size=None):
                    data_archive.write(chunk)

    def _unpack_data_archive(self: Self) -> None:
        if self._data_dir.is_dir():
            return
        shutil.unpack_archive(self._data_archive, extract_dir=self._data_dir)
