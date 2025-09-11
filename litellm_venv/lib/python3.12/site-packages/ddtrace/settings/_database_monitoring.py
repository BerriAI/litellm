from envier import En
from envier import validators


class DatabaseMonitoringConfig(En):
    __prefix__ = "dd_dbm"

    propagation_mode = En.v(
        str,
        "propagation_mode",
        default="disabled",
        help="Valid Injection Modes: disabled, service, and full",
        validator=validators.choice(["disabled", "full", "service"]),
    )


dbm_config = DatabaseMonitoringConfig()
