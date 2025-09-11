# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ------------------------------------
from enum import Enum
from azure.core import CaseInsensitiveEnumMeta


class RegionalAuthority(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Identifies a regional authority for authentication"""

    #: Attempt to discover the appropriate authority. This works on some Azure hosts, such as VMs and
    #: Azure Functions. The non-regional authority is used when discovery fails.
    AUTO_DISCOVER_REGION = "tryautodetect"

    ASIA_EAST = "eastasia"
    ASIA_SOUTHEAST = "southeastasia"
    AUSTRALIA_CENTRAL = "australiacentral"
    AUSTRALIA_CENTRAL_2 = "australiacentral2"
    AUSTRALIA_EAST = "australiaeast"
    AUSTRALIA_SOUTHEAST = "australiasoutheast"
    BRAZIL_SOUTH = "brazilsouth"
    CANADA_CENTRAL = "canadacentral"
    CANADA_EAST = "canadaeast"
    CHINA_EAST = "chinaeast"
    CHINA_EAST_2 = "chinaeast2"
    CHINA_NORTH = "chinanorth"
    CHINA_NORTH_2 = "chinanorth2"
    EUROPE_NORTH = "northeurope"
    EUROPE_WEST = "westeurope"
    FRANCE_CENTRAL = "francecentral"
    FRANCE_SOUTH = "francesouth"
    GERMANY_CENTRAL = "germanycentral"
    GERMANY_NORTH = "germanynorth"
    GERMANY_NORTHEAST = "germanynortheast"
    GERMANY_WEST_CENTRAL = "germanywestcentral"
    GOVERNMENT_US_ARIZONA = "usgovarizona"
    GOVERNMENT_US_DOD_CENTRAL = "usdodcentral"
    GOVERNMENT_US_DOD_EAST = "usdodeast"
    GOVERNMENT_US_IOWA = "usgoviowa"
    GOVERNMENT_US_TEXAS = "usgovtexas"
    GOVERNMENT_US_VIRGINIA = "usgovvirginia"
    INDIA_CENTRAL = "centralindia"
    INDIA_SOUTH = "southindia"
    INDIA_WEST = "westindia"
    JAPAN_EAST = "japaneast"
    JAPAN_WEST = "japanwest"
    KOREA_CENTRAL = "koreacentral"
    KOREA_SOUTH = "koreasouth"
    NORWAY_EAST = "norwayeast"
    NORWAY_WEST = "norwaywest"
    SOUTH_AFRICA_NORTH = "southafricanorth"
    SOUTH_AFRICA_WEST = "southafricawest"
    SWITZERLAND_NORTH = "switzerlandnorth"
    SWITZERLAND_WEST = "switzerlandwest"
    UAE_CENTRAL = "uaecentral"
    UAE_NORTH = "uaenorth"
    UK_SOUTH = "uksouth"
    UK_WEST = "ukwest"
    US_CENTRAL = "centralus"
    US_EAST = "eastus"
    US_EAST_2 = "eastus2"
    US_NORTH_CENTRAL = "northcentralus"
    US_SOUTH_CENTRAL = "southcentralus"
    US_WEST = "westus"
    US_WEST_2 = "westus2"
    US_WEST_CENTRAL = "westcentralus"
