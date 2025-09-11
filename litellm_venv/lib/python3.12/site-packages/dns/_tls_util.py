# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

import os
from typing import Tuple


def convert_verify_to_cafile_and_capath(
    verify: bool | str,
) -> Tuple[str | None, str | None]:
    cafile: str | None = None
    capath: str | None = None
    if isinstance(verify, str):
        if os.path.isfile(verify):
            cafile = verify
        elif os.path.isdir(verify):
            capath = verify
        else:
            raise ValueError("invalid verify string")
    return cafile, capath
