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

SPLITS = ("train", "validation", "test")
PROCESSES = 20


@dataclasses.dataclass(slots=True)
class DataLoader(ABC):
    """Data loader.

    Attributes:
        data_url: The url of the data.
        train_data_files: The relative paths to the files containing the train data.
        validation_data_files: The relative paths to the files containing the validation
          data.
        test_data_files: The relative paths to the files containing the test data.
        premise_column: The name of the premise column.
        hypothesis_column: The name of the hypothesis column.
        label_column: The name of the label column.
    """

    data_url: str
    train_data_files: str | Sequence[str] | None = None
    validation_data_files: str | Sequence[str] | None = None
    test_data_files: str | Sequence[str] | None = None
    premise_column: str = "premise"
    hypothesis_column: str = "hypothesis"
    label_column: str = "label"

    def __post_init__(self: Self) -> None:
        """Post init."""
        if (
            self.train_data_files is None
            and self.validation_data_files is None
            and self.test_data_files is None
        ):
            msg = "At least one of train, validation or test must be specified."
            raise ValueError(msg)

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
        return {
            split: (
                str(self._data_dir.joinpath(paths))
                if isinstance(paths, str)
                else [str(self._data_dir.joinpath(path)) for path in paths]
            )
            for split in SPLITS
            if (paths := getattr(self, split)) is not None
        }

    @property
    def _data_files_suffix(self: Self) -> str:
        path = list(self._data_files.values())[0]
        if isinstance(path, list):
            path = path[0]
        return PurePath(path).suffix

    @property
    def _data_files_type(self: Self) -> str:
        match self._data_files_suffix:
            case ".csv" | ".tsv":
                return "csv"
            case ".json" | ".jsonl":
                return "json"
            case _:
                raise ValueError("Unknown file suffix: " + self._data_files_suffix)

    @property
    def _data_files_delimiter(self: Self) -> str | None:
        match self._data_files_suffix:
            case ".csv":
                return ","
            case ".tsv":
                return "\t"
            case _:
                return None

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
        pass

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

    def _load_data_files(self: Self, split: str | None = None) -> Dataset | DatasetDict:
        config_kwargs = {}
        if self._data_files_delimiter is not None:
            config_kwargs = {
                "delimiter": self._data_files_delimiter,
                "quoting": csv.QUOTE_NONE,
            }
        return (
            datasets.load_dataset(
                self._data_files_type,
                data_files=self._data_files,
                split=split,
                **config_kwargs,
            )
            .rename_columns(self._column_mapping)
            .select_columns(self._columns)
            .filter(
                lambda label: label in self._labels,
                input_columns="label",
                num_proc=PROCESSES,
                desc="Filtering NA labels",
            )
            .map(
                lambda label: {"label": self._label_mapping[label]},
                input_columns="label",
                num_proc=PROCESSES,
                desc="Renaming labels",
            )
            .class_encode_column("label")
        )

    def _load_data(self: Self, split: str | None = None) -> Dataset | DatasetDict:
        self._create_data_cache_dir()
        self._download_data_archive()
        self._unpack_data_archive()
        return self._load_data_files(split=split)

    def load_dataset(self: Self, split: str) -> Dataset:
        """Load split of the data.

        Args:
            split: The name of the split.
        """
        return self._load_data(split=split)

    def load_dataset_dict(self: Self) -> DatasetDict:
        """Load the data."""
        return self._load_data()


@dataclasses.dataclass(slots=True)
class RTEDataLoader(DataLoader):
    """RTE data loader.

    Attributes:
        entailment_label: The label of the entailment class.
        not_entailment_label: The label of the not_entailment class.
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
class NLIDataLoader(DataLoader):
    """NLI data loader.

    Attributes:
        entailment_label: The label of the entailment class.
        neutral_label: The label of the neutral class.
        contradiction_label: The label of the contradiction class.
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


if __name__ == "__main__":
    print(
        NLIDataLoader(
            "https://nlp.stanford.edu/projects/snli/snli_1.0.zip",
            train="snli_1.0/snli_1.0_train.jsonl",
            validation="snli_1.0/snli_1.0_dev.jsonl",
            test="snli_1.0/snli_1.0_test.jsonl",
            premise_column="sentence1",
            hypothesis_column="sentence2",
            label_column="gold_label",
        ).load_dataset_dict()["validation"][66]
    )
