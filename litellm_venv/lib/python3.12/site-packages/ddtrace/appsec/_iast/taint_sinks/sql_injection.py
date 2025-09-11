from .. import oce
from ..constants import VULN_SQL_INJECTION
from ._base import VulnerabilityBase


@oce.register
class SqlInjection(VulnerabilityBase):
    vulnerability_type = VULN_SQL_INJECTION
