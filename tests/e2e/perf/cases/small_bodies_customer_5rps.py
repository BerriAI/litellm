from dataclasses import replace
from typing import Final

from cases.small_bodies_5rps import CASE as BASELINE

CASE: Final = replace(BASELINE, name="small_bodies_customer_5rps", customer_hierarchy=True)
