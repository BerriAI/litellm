from typing import Dict
from typing import Text
from typing import Union

from ddtrace.internal.compat import NumericType


_TagNameType = Union[Text, bytes]
_MetaDictType = Dict[_TagNameType, Text]
_MetricDictType = Dict[_TagNameType, NumericType]
