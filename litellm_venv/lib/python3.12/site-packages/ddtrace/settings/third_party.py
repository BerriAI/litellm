from envier import En


class ThirdPartyDetectionConfig(En):
    __prefix__ = "dd.third_party_detection"

    excludes = En.v(
        set,
        "excludes",
        help="List of packages that should not be treated as third-party",
        help_type="List",
        default=set(),
    )
    includes = En.v(
        set,
        "includes",
        help="Additional packages to treat as third-party",
        help_type="List",
        default=set(),
    )


config = ThirdPartyDetectionConfig()
