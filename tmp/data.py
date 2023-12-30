import collections
import itertools

DATASETS_LOADERS = {
    "snli": NLIDatasetLoader(
        "https://nlp.stanford.edu/projects/snli/snli_1.0.zip",
        train="snli_1.0/snli_1.0_train.jsonl",
        validation="snli_1.0/snli_1.0_dev.jsonl",
        test="snli_1.0/snli_1.0_test.jsonl",
        premise="sentence1",
        hypothesis="sentence2",
        label="gold_label",
        na_value="-",
    ),
    "multinli": NLIDatasetLoader(
        "https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip",
        train="multinli_1.0/multinli_1.0_train.jsonl",
        validation=[
            "multinli_1.0/multinli_1.0_dev_matched.jsonl",
            "multinli_1.0/multinli_1.0_dev_mismatched.jsonl",
        ],
        premise="sentence1",
        hypothesis="sentence2",
        label="gold_label",
        na_value="-",
    ),
    "multinli_matched": NLIDatasetLoader(
        "https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip",
        train="multinli_1.0/multinli_1.0_train.jsonl",
        validation="multinli_1.0/multinli_1.0_dev_matched.jsonl",
        premise="sentence1",
        hypothesis="sentence2",
        label="gold_label",
        na_value="-",
    ),
    "multinli_mismatched": NLIDatasetLoader(
        "https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip",
        train="multinli_1.0/multinli_1.0_train.jsonl",
        validation="multinli_1.0/multinli_1.0_dev_mismatched.jsonl",
        premise="sentence1",
        hypothesis="sentence2",
        label="gold_label",
        na_value="-",
    ),
    "glue/rte": RTEDatasetLoader(
        "https://dl.fbaipublicfiles.com/glue/data/RTE.zip",
        train="RTE/train.tsv",
        validation="RTE/dev.tsv",
        premise="sentence1",
        hypothesis="sentence2",
    ),
    "glue/qnli": RTEDatasetLoader(
        "https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip",
        train="QNLI/train.tsv",
        validation="QNLI/dev.tsv",
        premise="question",
        hypothesis="sentence",
    ),
    "glue/wnli": RTEDatasetLoader(
        "https://dl.fbaipublicfiles.com/glue/data/WNLI.zip",
        train="WNLI/train.tsv",
        validation="WNLI/dev.tsv",
        premise="sentence1",
        hypothesis="sentence2",
        entailment=1,
        not_entailment=0,
    ),
    "superglue/cb": NLIDatasetLoader(
        "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/CB.zip",
        train="CB/train.jsonl",
        validation="CB/val.jsonl",
    ),
    "anli/r1": NLIDatasetLoader(
        "https://dl.fbaipublicfiles.com/anli/anli_v0.1.zip",
        train="anli_v0.1/R1/train.jsonl",
        validation="anli_v0.1/R1/dev.jsonl",
        test="anli_v0.1/R1/test.jsonl",
        premise="context",
        entailment="e",
        neutral="n",
        contradiction="c",
    ),
    "anli/r2": NLIDatasetLoader(
        "https://dl.fbaipublicfiles.com/anli/anli_v0.1.zip",
        train="anli_v0.1/R2/train.jsonl",
        validation="anli_v0.1/R2/dev.jsonl",
        test="anli_v0.1/R2/test.jsonl",
        premise="context",
        entailment="e",
        neutral="n",
        contradiction="c",
    ),
    "anli/r3": NLIDatasetLoader(
        "https://dl.fbaipublicfiles.com/anli/anli_v0.1.zip",
        train="anli_v0.1/R3/train.jsonl",
        validation="anli_v0.1/R3/dev.jsonl",
        test="anli_v0.1/R3/test.jsonl",
        premise="context",
        entailment="e",
        neutral="n",
        contradiction="c",
    ),
    "scitail": RTEDatasetLoader(
        "https://ai2-public-datasets.s3.amazonaws.com/scitail/SciTailV1.1.zip",
        train="SciTailV1.1/predictor_format/scitail_1.0_structure_train.jsonl",
        validation="SciTailV1.1/predictor_format/scitail_1.0_structure_dev.jsonl",
        test="SciTailV1.1/predictor_format/scitail_1.0_structure_test.jsonl",
        premise="sentence1",
        hypothesis="sentence2",
        label="gold_label",
        entailment="entails",
        not_entailment="neutral",
    ),
}


# Check that the DataLoaders are the same type

@dataclasses.dataclass(slots=True)
class DatasetCombiner:
    """Combine multiple datasets into a single dataset."""

    names: list[str]

    def __post_init__(self: Self) -> None:
        """Validate the dataset names."""
        for name in self.names:
            if name not in DATASETS_LOADERS:
                raise ValueError("Unknown dataset: " + name)

    @property
    def _dataset_loaders(self: Self) -> list[DatasetLoader]:
        return [DATASETS_LOADERS[name] for name in self.names]

    def load_dataset(self: Self, splits: str | list[str]) -> Dataset:
        """Load a split of the dataset."""
        if isinstance(splits, str):
            splits = list(itertools.repeat(splits, len(self._dataset_loaders)))
        return datasets.concatenate_datasets(
            [
                dataset_loader.load_dataset(split)
                for dataset_loader, split in zip(
                    self._dataset_loaders, splits, strict=True
                )
            ]
        )

    def load_dataset_dict(self: Self) -> DatasetDict:
        """Load the dataset."""
        dataset_dict = collections.defaultdict(list)
        for dataset_loader in self._dataset_loaders:
            for split, dataset in dataset_loader.load_dataset_dict().items():
                dataset_dict[split].append(dataset)
        return DatasetDict(
            {
                split: datasets.concatenate_datasets(_datasets)
                for split, _datasets in dataset_dict.items()
            }
        )


DATASET_COMBINERS = {"anli": DatasetCombiner(["anli/r1", "anli/r2", "anli/r3"])}


@dataclasses.dataclass(slots=True)
class MetaLoader:
    """Load a dataset or a combination of datasets."""

    names: list[str]

    def __init__(self: Self, *names: str) -> None:
        """Initialize the loader."""
        self.names = []
        for name in names:
            if name in DATASETS_LOADERS:
                self.names.append(name)
            elif name in DATASET_COMBINERS:
                self.names.extend(DATASET_COMBINERS[name].names)
            else:
                raise ValueError("Unknown dataset: " + name)

    def _dataset_combiner(self: Self) -> DatasetCombiner:
        return DatasetCombiner(self.names)

    def load_dataset(self: Self, splits: str | list[str]) -> Dataset:
        """Load the dataset."""
        if isinstance(splits, list) and len(self.names) != len(splits):
            msg = "Must provide either a single split or a split for each dataset."
            raise ValueError(msg)
        return self._dataset_combiner().load_dataset(splits)

    def load_dataset_dict(self: Self) -> DatasetDict:
        """Load the dataset."""
        return self._dataset_combiner().load_dataset_dict()


if __name__ == "__main__":
    print(MetaLoader("glue/rte", "glue/wnli", "scitail").load_dataset_dict())
    print(MetaLoader("anli", "multinli", "snli", "superglue/cb").load_dataset_dict())
