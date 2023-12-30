import dataclasses

from datasets import ClassLabel, Features, Sequence, Value


@dataclasses.dataclass
class DatasetInfo:
    path: str
    name: str | None
    text_key: str
    text_pair_key: str
    label_key: str
    train_dataset_keys: list[str] | None
    eval_dataset_keys: list[str] | None
    test_dataset_keys: list[str] | None
    features: Features | None = None


DATASET_INFO = {
    "esnli": DatasetInfo(
        path="esnli",
        name=None,
        text_key="premise",
        text_pair_key="hypothesis",
        label_key="label",
        train_dataset_keys=["train"],
        eval_dataset_keys=["validation"],
        test_dataset_keys=["test"],
    ),
    "adv_multinli_matched": DatasetInfo(
        path="adv_glue",
        name="adv_mnli",
        text_key="premise",
        text_pair_key="hypothesis",
        label_key="label",
        train_dataset_keys=None,
        eval_dataset_keys=["validation"],
        test_dataset_keys=None,
    ),
    "adv_multinli_mismatched": DatasetInfo(
        path="adv_glue",
        name="adv_mnli_mismatched",
        text_key="premise",
        text_pair_key="hypothesis",
        label_key="label",
        train_dataset_keys=None,
        eval_dataset_keys=["validation"],
        test_dataset_keys=None,
    ),
    "mrpc": DatasetInfo(
        path="glue",
        name="mrpc",
        text_key="sentence1",
        text_pair_key="sentence2",
        label_key="label",
        train_dataset_keys=["train"],
        eval_dataset_keys=["validation"],
        test_dataset_keys=["test"],
    ),
    "qnli": DatasetInfo(
        path="glue",
        name="qnli",
        text_key="question",
        text_pair_key="sentence",
        label_key="label",
        train_dataset_keys=["train"],
        eval_dataset_keys=["validation"],
        test_dataset_keys=None,
    ),
    "adv_qnli": DatasetInfo(
        path="adv_glue",
        name="adv_qnli",
        text_key="question",
        text_pair_key="sentence",
        label_key="label",
        train_dataset_keys=None,
        eval_dataset_keys=["validation"],
        test_dataset_keys=None,
    ),
    "qqp": DatasetInfo(
        path="glue",
        name="qqp",
        text_key="question1",
        text_pair_key="question2",
        label_key="label",
        train_dataset_keys=["train"],
        eval_dataset_keys=["validation"],
        test_dataset_keys=None,
    ),
    "adv_qqp": DatasetInfo(
        path="adv_glue",
        name="adv_qqp",
        text_key="question1",
        text_pair_key="question2",
        label_key="label",
        train_dataset_keys=None,
        eval_dataset_keys=["validation"],
        test_dataset_keys=None,
    ),
    "wnli": DatasetInfo(
        path="glue",
        name="wnli",
        text_key="sentence1",
        text_pair_key="sentence2",
        label_key="label",
        train_dataset_keys=["train"],
        eval_dataset_keys=["validation"],
        test_dataset_keys=None,
    ),
    "diagnostic": DatasetInfo(
        path="super_glue",
        name="axb",
        text_key="sentence1",
        text_pair_key="sentence2",
        label_key="label",
        train_dataset_keys=None,
        eval_dataset_keys=None,
        test_dataset_keys=["test"],
    ),
    "adv_rte": DatasetInfo(
        path="adv_glue",
        name="adv_rte",
        text_key="sentence1",
        text_pair_key="sentence2",
        label_key="label",
        train_dataset_keys=None,
        eval_dataset_keys=["validation"],
        test_dataset_keys=None,
    ),
    "winogender": DatasetInfo(
        path="super_glue",
        name="axg",
        text_key="premise",
        text_pair_key="hypothesis",
        label_key="label",
        train_dataset_keys=None,
        eval_dataset_keys=None,
        test_dataset_keys=["test"],
    ),
    "hans": DatasetInfo(
        path="hans",
        name=None,
        text_key="premise",
        text_pair_key="hypothesis",
        label_key="label",
        train_dataset_keys=["train"],
        eval_dataset_keys=["validation"],
        test_dataset_keys=None,
    ),
    "sem_eval_2014_task_1": DatasetInfo(
        path="sem_eval_2014_task_1",
        name=None,
        text_key="premise",
        text_pair_key="hypothesis",
        label_key="entailment_judgment",
        train_dataset_keys=["train"],
        eval_dataset_keys=["validation"],
        test_dataset_keys=["test"],
    ),
    "sick": DatasetInfo(
        path="sick",
        name=None,
        text_key="sentence_A",
        text_pair_key="sentence_B",
        label_key="label",
        train_dataset_keys=["train"],
        eval_dataset_keys=["validation"],
        test_dataset_keys=["test"],
    ),
}

DATASETS = list(DATASET_INFO.keys())
