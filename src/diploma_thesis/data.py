"""Load data."""

import abc
import csv
import dataclasses
import shutil
from abc import ABC
from collections.abc import Sequence
from pathlib import Path, PurePath
from typing import Self
from urllib import parse

import datasets
import requests
from datasets import Dataset, DatasetDict
from tqdm import tqdm

DIPLOMA_THESIS_CACHE = Path("~/.cache/diploma-thesis/").expanduser()
DATA_CACHE = DIPLOMA_THESIS_CACHE.joinpath("data/")

TIMEOUT = (3.05, 27)
CHUNK_SIZE = 4096


@dataclasses.dataclass(slots=True)
class DataLoader(ABC):
    """Data loader.

    Attributes:
        data_url: The url of the data.
        premise_column: The name of the premise column.
        hypothesis_column: The name of the hypothesis column.
        label_column: The name of the label column.
    """

    data_url: str
    premise_column: str = "premise"
    hypothesis_column: str = "hypothesis"
    label_column: str = "label"

    def __post_init__(self: Self) -> None:

    @property
    def _data_url_path(self: Self) -> PurePath:
        return PurePath(parse.urlparse(self.data_url).path)

    @property
    def _data_archive(self: Self) -> Path:
        return DATA_CACHE.joinpath(self._data_url_path.name)

    @property
    def _data_dir(self: Self) -> Path:
        return DATA_CACHE.joinpath(self._data_url_path.stem)

    @property
    def _data_files(self: Self) -> dict[str, str | list[str]]:
    @property
    def _column_mapping(self: Self) -> dict[str, str]:
        return {
            self.premise_column: "premise",
            self.hypothesis_column: "hypothesis",
            self.label_column: "label",
        }

    @property
    def _columns(self: Self) -> list[str]:
        return list(self._column_mapping.values())

    @property
    @abc.abstractmethod
    def _label_mapping(self: Self) -> dict[str | int, str]:

    @property
    def _labels(self: Self) -> list[str]:
        return list(self._label_mapping.values())

    @staticmethod
    def _create_data_cache_dir() -> None:
        DATA_CACHE.mkdir(parents=True, exist_ok=True)

    def _download_data_archive(self: Self) -> None:
        if self._data_archive.is_file():
            return
        with requests.get(self.data_url, timeout=TIMEOUT, stream=True) as response:
            response.raise_for_status()
            with tqdm.wrapattr(
                self._data_archive.open(mode="bw"),
                "write",
                total=int(response.headers.get("content-length", 0)),
                desc="Downloading " + self._data_archive.name,
                miniters=1,
            ) as data_archive:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    data_archive.write(chunk)

    def _unpack_data_archive(self: Self) -> None:
        if self._data_dir.is_dir():
            return
        shutil.unpack_archive(self._data_archive, extract_dir=self._data_dir)

        return (
            datasets.load_dataset(
                data_files=self._data_files,
            .rename_columns(self._column_mapping)
            .class_encode_column("label")
        )

    def _load_data(self: Self, split: str | None = None) -> Dataset | DatasetDict:
        self._create_data_cache_dir()
        self._download_data_archive()
        self._unpack_data_archive()
        return self._load_data_files(split=split)

    def load_dataset(self: Self, split: str) -> Dataset:

        Args:
        """
        return self._load_data(split=split)

    def load_dataset_dict(self: Self) -> DatasetDict:
        return self._load_data()


@dataclasses.dataclass(slots=True)

    Attributes:
    """

    entailment_label: str | int = "entailment"
    not_entailment_label: str | int = "not_entailment"

    @property
    def _label_mapping(self: Self) -> dict[str | int, str]:
        return {
            self.entailment_label: "entailment",
            self.not_entailment_label: "not_entailment",
        }


@dataclasses.dataclass(slots=True)

    Attributes:
    """

    entailment_label: str | int = "entailment"
    neutral_label: str | int = "neutral"
    contradiction_label: str | int = "contradiction"

    @property
    def _label_mapping(self: Self) -> dict[str | int, str]:
        return {
            self.entailment_label: "entailment",
            self.neutral_label: "neutral",
            self.contradiction_label: "contradiction",
        }
