"""PyVizier classes for Pythia policies."""

try:
    from vizier.pyvizier import MetricInformation
    from vizier.pyvizier import MetricsConfig
    from vizier.pyvizier import MetricType
    from vizier.pyvizier import (
        ObjectiveMetricGoal,
    )
    from vizier.pyvizier import ProblemStatement
    from vizier.pyvizier import SearchSpace
    from vizier.pyvizier import (
        SearchSpaceSelector,
    )
    from vizier.pyvizier import Metadata
    from vizier.pyvizier import MetadataValue
    from vizier.pyvizier import Namespace
    from vizier.pyvizier import ExternalType
    from vizier.pyvizier import ParameterConfig
    from vizier.pyvizier import ParameterType
    from vizier.pyvizier import ScaleType
    from vizier.pyvizier import CompletedTrial
    from vizier.pyvizier import Measurement
    from vizier.pyvizier import MonotypeParameterSequence
    from vizier.pyvizier import Metric
    from vizier.pyvizier import ParameterDict
    from vizier.pyvizier import ParameterValue
    from vizier.pyvizier import Trial
    from vizier.pyvizier import ParameterValueTypes
    from vizier.pyvizier import TrialFilter
    from vizier.pyvizier import TrialStatus
    from vizier.pyvizier import TrialSuggestion
except ImportError:
    raise ImportError(
        "Google-vizier is not installed, and is required to use Vizier client."
        'Please install the SDK using "pip install google-vizier"'
    )

from google.cloud.aiplatform.vizier.pyvizier.proto_converters import TrialConverter
from google.cloud.aiplatform.vizier.pyvizier.proto_converters import (
    ParameterConfigConverter,
)
from google.cloud.aiplatform.vizier.pyvizier.proto_converters import (
    MeasurementConverter,
)
from google.cloud.aiplatform.vizier.pyvizier.study_config import StudyConfig
from google.cloud.aiplatform.vizier.pyvizier.study_config import Algorithm
from google.cloud.aiplatform.vizier.pyvizier.automated_stopping import (
    AutomatedStoppingConfig,
)

__all__ = (
    "MetricInformation",
    "MetricsConfig",
    "MetricType",
    "ObjectiveMetricGoal",
    "ProblemStatement",
    "SearchSpace",
    "SearchSpaceSelector",
    "Metadata",
    "MetadataValue",
    "Namespace",
    "ParameterConfigConverter",
    "ParameterValueTypes",
    "MeasurementConverter",
    "MonotypeParameterSequence",
    "TrialConverter",
    "StudyConfig",
    "Algorithm",
    "AutomatedStoppingConfig",
    "ExternalType",
    "ParameterConfig",
    "ParameterType",
    "ScaleType",
    "CompletedTrial",
    "Measurement",
    "Metric",
    "ParameterDict",
    "ParameterValue",
    "Trial",
    "TrialFilter",
    "TrialStatus",
    "TrialSuggestion",
)
