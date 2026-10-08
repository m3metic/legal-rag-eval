from .maud import MAUD
from .obliqa import ObliQA

DATASETS = {"obliqa": ObliQA, "maud": MAUD}


def load_dataset(name: str, **kwargs):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Options: {sorted(DATASETS)}")
    return DATASETS[name](**kwargs)
