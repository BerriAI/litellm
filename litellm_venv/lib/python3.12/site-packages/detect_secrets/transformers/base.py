from abc import ABCMeta
from abc import abstractmethod
from typing import List

from ..types import NamedIO


class BaseTransformer(metaclass=ABCMeta):
    """
    There are special filetypes (e.g. YAML) that work better with our line-based secrets parsing
    if we parse the file differently. In these cases, transformers can take the file, and parse
    it to meet the needs of the secret detector.

    While the transformation may not be an original copy, it just needs to proxy the original
    file so that we can obtain:
        1. The secret value
        2. The specific line that it's found on (for auditing purposes)
    """

    @property
    def is_eager(self) -> bool:
        """
        Eager transformers tend to be over-aggressive, and cause performance issues / false
        positives. We can make a transformer less eager through stricter validation checks
        on `should_parse_file`, however, in the cases where we are unable to do so, this flag
        informs the scanner to only use this transformer if all other methods fail to obtain
        secrets.
        """
        return False

    @abstractmethod
    def should_parse_file(self, filename: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def parse_file(self, file: NamedIO) -> List[str]:
        """
        :raises: ParsingError
        """
        raise NotImplementedError
