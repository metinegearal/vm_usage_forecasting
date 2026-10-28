"""Read the two release files; construct full targets before splitting."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

CADENCE = 300
CATEGORIES = ["Delay-insensitive", "Interactive", "Unknown"]
BUCKETS = ["1-3", "4-20", ">20"]
TARGETS = ["active_vm_count", "aggregate_cpu_utilization"]

@dataclass
class Dataset:
    cpu: pd.DataFrame
    targets: pd.DataFrame
    metadata: pd.DataFrame
    groups: pd.MultiIndex
    group_id: np.ndarray
    audit: dict

    def group_means(self, end):
        values = self.cpu.iloc[:end].to_numpy()
        means = np.full((end, len(self.groups)), np.nan)
        for g in range(len(self.groups)):
            selected = values[:, self.group_id == g]
            if selected.shape[1]:
                counts = np.isfinite(selected).sum(axis=1)
                means[:, g] = np.divide(np.nansum(selected, axis=1), counts,
                    out=np.full(end, np.nan), where=counts > 0)
        return means

    def counts_at(self, end):
        valid = self.cpu.iloc[end-1].notna().to_numpy()
        return np.bincount(self.group_id[valid], minlength=len(self.groups))


def source_file(directory, stem):
    for suffix in (".csv.gz", ".csv"):
        path = Path(directory) / (stem + suffix)
        if path.is_file():
            return path
    raise FileNotFoundError(f"Need {stem}.csv.gz (or .csv) inside {directory}")


def load_data(directory):
    return build_data(pd.read_csv(source_file(directory, "vm_cpu_readings")),
                      pd.read_csv(source_file(directory, "vm_metadata")))


def build_data(readings, metadata):
    r, m = readings.copy(), metadata.copy()
    for name, frame, columns in [("readings", r, ["timestamp", "vm_id", "avg_cpu"]),
                                  ("metadata", m, ["vm_id", "subscription_id", "vm_category"])]:
        missing = set(columns) - set(frame)
        if missing:
            raise ValueError(f"{name}: missing columns {sorted(missing)}")
    audit = dict(source_reading_rows=len(r), source_metadata_rows=len(m),
        reading_missing_cells=int(r.isna().sum().sum()),
        metadata_missing_cells=int(m.isna().sum().sum()))
    for frame in (r, m):
        frame["vm_id"] = frame.vm_id.astype("string").str.strip().replace("", pd.NA)
    if r.vm_id.isna().any() or m.vm_id.isna().any():
        raise ValueError("Missing VM identifiers must be resolved explicitly")
    if m.vm_id.duplicated().any():
        raise ValueError("Metadata must contain exactly one row per VM")
    r["timestamp"] = pd.to_numeric(r.timestamp, errors="raise")
    t = r.timestamp.to_numpy()
    if not len(t) or not (np.isfinite(t) & (t >= 0) & (t % 1 == 0)).all():
        raise ValueError("Timestamps must be finite nonnegative integer seconds")
    r["timestamp"] = r.timestamp.astype("int64")
    observed = np.sort(r.timestamp.unique())
    if ((observed - observed[0]) % CADENCE).any():
        raise ValueError("Timestamps must follow a 300-second grid")
    r["avg_cpu"] = pd.to_numeric(r.avg_cpu, errors="coerce")
    if {"min_cpu", "max_cpu"}.issubset(r):
        lower = pd.to_numeric(r.min_cpu, errors="coerce")
        upper = pd.to_numeric(r.max_cpu, errors="coerce")
        coherent = np.isfinite(lower) & np.isfinite(upper) & lower.ge(0) & upper.le(100) & lower.le(r.avg_cpu) & r.avg_cpu.le(upper)
        audit["incoherent_interval_statistics_rows"] = int((~coherent).sum())
    else:
        audit["incoherent_interval_statistics_rows"] = None
    # Collapse exact target duplicates only; never average conflicting VM keys.
    exact = int(r.duplicated(["vm_id", "timestamp", "avg_cpu"]).sum())
    r = r.drop_duplicates(["vm_id", "timestamp", "avg_cpu"])
    if r.duplicated(["vm_id", "timestamp"]).any():
        raise ValueError("Conflicting duplicate (vm_id, timestamp) CPU values")
    grid = pd.Index(np.arange(observed[0], observed[-1]+CADENCE, CADENCE), name="timestamp")
    valid = np.isfinite(r.avg_cpu) & r.avg_cpu.between(0, 100)
    clean = r.loc[valid, ["timestamp", "vm_id", "avg_cpu"]]
    ids = pd.Index(sorted(r.vm_id.unique()), name="vm_id")
    m = m.set_index("vm_id")
    if not ids.isin(m.index).all():
        raise ValueError("Every observed VM needs a metadata record")
    # Metadata supplied for the challenge population is known at the origin.
    # Counts describe this supplied sample, not a customer's full Azure estate.
    subscriptions = m.subscription_id.astype("string").str.strip().replace("", pd.NA)
    m["subscription_group_size"] = subscriptions.map(subscriptions.value_counts()).fillna(1).astype(int)
    m["vm_category"] = m.vm_category.fillna("Unknown").astype(str)
    unknown = ~m.vm_category.isin(CATEGORIES)
    audit["unrecognized_categories_mapped_to_unknown"] = int(unknown.sum())
    m.loc[unknown, "vm_category"] = "Unknown"
    m["subscription_size"] = pd.cut(m.subscription_group_size, [0, 3, 20, np.inf], labels=BUCKETS)
    m = m.reindex(ids)
    types = pd.Categorical(m.vm_category, categories=CATEGORIES).codes
    sizes = pd.Categorical(m.subscription_size, categories=BUCKETS).codes
    group_id = types*3 + sizes
    groups = pd.MultiIndex.from_product([CATEGORIES, BUCKETS], names=["vm_type", "subscription_size"])
    cpu = clean.pivot(index="timestamp", columns="vm_id", values="avg_cpu").reindex(index=grid, columns=ids)
    targets = pd.DataFrame({"active_vm_count": cpu.notna().sum(axis=1),
                           "aggregate_cpu_utilization": cpu.mean(axis=1)}, index=grid)
    audit.update(exact_target_duplicates_removed=exact, invalid_avg_cpu_rows_excluded=int((~valid).sum()),
        unique_vms=len(ids), grid_timestamps=len(grid), source_timestamps=len(observed),
        wholly_missing_source_timestamps=int(len(grid)-len(observed)),
        wholly_missing_cpu_timestamps=int(targets.aggregate_cpu_utilization.isna().sum()),
        missing_vm_timestamp_readings=int(cpu.isna().sum().sum()), cadence_seconds=CADENCE,
        start_timestamp=int(grid[0]), end_timestamp=int(grid[-1]))
    # Source metadata CPU summaries/deletion times are not forecasting inputs.
    return Dataset(cpu, targets, m, groups, group_id, audit)


def split(data, horizon=24, train_steps=None):
    if horizon < 1:
        raise ValueError("horizon must be positive")
    n = len(data.targets)-horizon if train_steps is None else train_steps
    if n < 2 or n+horizon != len(data.targets):
        raise ValueError("Need at least two training timestamps and train_steps + horizon == full grid length")
    if data.audit["wholly_missing_source_timestamps"]:
        raise ValueError("Whole-timestamp source gaps: targets retain NaN CPU; resolve telemetry rather than inventing labels")
    if data.targets.iloc[:n].aggregate_cpu_utilization.isna().any():
        raise ValueError("Training has whole-timestamp CPU gaps; no CPU interpolation is performed")
    return n, data.targets.index[n:]


def validate_predictions(frame, expected_times):
    if frame.columns.tolist() != ["timestamp", *TARGETS]:
        raise ValueError("Submission columns must be timestamp, active_vm_count, aggregate_cpu_utilization")
    if not np.array_equal(frame.timestamp.to_numpy(), np.asarray(expected_times)) or frame.timestamp.duplicated().any():
        raise ValueError("Submission timestamps must match all holdout timestamps exactly, in order")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("All submission fields must be finite numeric values")
    if (frame.active_vm_count < 0).any() or not frame.aggregate_cpu_utilization.between(0,100).all():
        raise ValueError("Invalid count or CPU prediction range")
