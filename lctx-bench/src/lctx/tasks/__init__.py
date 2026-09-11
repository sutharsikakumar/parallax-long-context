"""Task suite. Importing this package registers every built-in task."""

from .base import (  # noqa: F401
    ABSENT,
    Needle,
    Task,
    TaskInstance,
    TaskSpec,
    all_tasks,
    get_task,
    register_task,
)
from . import (  # noqa: F401
    aggregation,
    multi_hop,
    multi_needle,
    negative_retrieval,
    ordering,
    selective_copy,
    single_needle,
)

__all__ = [
    "ABSENT", "Needle", "Task", "TaskInstance", "TaskSpec",
    "all_tasks", "get_task", "register_task",
]
