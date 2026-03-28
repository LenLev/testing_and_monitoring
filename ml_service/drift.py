import asyncio
import importlib
import logging
import os
from dataclasses import dataclass, field

import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass
class DriftBuffer:
    reference: pd.DataFrame | None = None
    current_chunk: list[dict[str, object]] = field(default_factory=list)

    def add(self, row: dict[str, object]) -> None:
        self.current_chunk.append(row)

    def flush_current(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.current_chunk)
        self.current_chunk = []
        return frame


DRIFT_BUFFER = DriftBuffer()


def drift_enabled() -> bool:
    return bool(os.getenv('EVIDENTLY_URL') and os.getenv('EVIDENTLY_PROJECT_ID'))


def track_for_drift(feature_values: dict[str, object], prediction: int, probability: float) -> None:
    DRIFT_BUFFER.add(
        {
            **feature_values,
            'prediction': prediction,
            'probability': probability,
        }
    )


async def run_drift_reporter() -> None:
    if not drift_enabled():
        LOGGER.info('Evidently reporting disabled: missing EVIDENTLY_URL or EVIDENTLY_PROJECT_ID')
        return

    try:
        evidently_module = importlib.import_module('evidently')
        presets_module = importlib.import_module('evidently.presets')
        workspace_module = importlib.import_module('evidently.ui.workspace')
        report_cls = evidently_module.Report
        drift_preset_cls = presets_module.DataDriftPreset
        remote_workspace_cls = workspace_module.RemoteWorkspace
    except ImportError:
        LOGGER.warning('Evidently package is not installed, drift reporting is disabled')
        return

    interval_seconds = int(os.getenv('DRIFT_INTERVAL_SECONDS', '300'))
    min_samples = int(os.getenv('DRIFT_MIN_SAMPLES', '100'))
    evidently_url = os.getenv('EVIDENTLY_URL')
    project_id = os.getenv('EVIDENTLY_PROJECT_ID')
    workspace = remote_workspace_cls(evidently_url)

    while True:
        await asyncio.sleep(interval_seconds)

        if len(DRIFT_BUFFER.current_chunk) < min_samples:
            continue

        current_data = DRIFT_BUFFER.flush_current()
        if DRIFT_BUFFER.reference is None:
            DRIFT_BUFFER.reference = current_data.copy()
            LOGGER.info('Initialized drift reference data with %s rows', len(current_data))
            continue

        try:
            drift_report = report_cls(metrics=[drift_preset_cls()])
            result = drift_report.run(
                reference_data=DRIFT_BUFFER.reference,
                current_data=current_data,
            )
            workspace.add_run(project_id, result)
            LOGGER.info('Uploaded drift report to Evidently, rows=%s', len(current_data))
        except Exception:
            LOGGER.exception('Failed to build or upload drift report')
