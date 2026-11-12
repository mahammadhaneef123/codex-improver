"""CSV persistence and deterministic dataset splitting."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

from .schema import CSV_HEADERS, MaintenanceRecord


@dataclass(slots=True)
class DatasetSplit:
    train: list[MaintenanceRecord]
    val: list[MaintenanceRecord]
    test: list[MaintenanceRecord]

    def __len__(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)


def write_csv(records: list[MaintenanceRecord], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: ("" if value is None else value)
                    for key, value in record.to_dict().items()
                }
            )
    return target


def read_csv(path: str | Path) -> list[MaintenanceRecord]:
    source = Path(path)
    with source.open("r", newline="", encoding="utf-8") as handle:
        return [MaintenanceRecord.from_dict(row) for row in csv.DictReader(handle)]


def split_records(
    records: list[MaintenanceRecord],
    train_split: float = 0.7,
    val_split: float = 0.15,
    *,
    seed: int = 42,
) -> DatasetSplit:
    if not records:
        raise ValueError("cannot split an empty dataset")
    if train_split <= 0 or val_split < 0 or train_split + val_split >= 1:
        raise ValueError("split ratios must leave room for a test set")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    train_end = int(len(shuffled) * train_split)
    val_end = train_end + int(len(shuffled) * val_split)

    return DatasetSplit(
        train=shuffled[:train_end],
        val=shuffled[train_end:val_end],
        test=shuffled[val_end:],
    )
