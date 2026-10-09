"""Compatibility entry point for the retired synchronous SPLAT workflow.

Use ``plctestbench.human_listening.assemble_study`` to export MUSHRA studies
and explicitly import responses. No webMUSHRA checkout or local results DB is
needed for preparation.
"""

from .human_listening import (
    ConditionDefinition,
    HumanCalculatorData,
    HumanResponse,
    ListeningStudy,
    ListeningTestDefinition,
    TrialDefinition,
    assemble_study,
)

__all__ = [
    "ConditionDefinition", "HumanCalculatorData", "HumanResponse",
    "ListeningStudy", "ListeningTestDefinition", "TrialDefinition", "assemble_study",
    "ListeningTest",
]


class ListeningTest:
    def __init__(self, settings) -> None:
        raise NotImplementedError(
            "The local-db SPLAT ListeningTest workflow is retired. "
            "Prepare HumanCalculator bundles and use assemble_study(...), "
            "study.export(...), and study.finalize(...) for MUSHRA quality scores."
        )
