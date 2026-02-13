from typing import Any, Dict


class CBFRecord(Dict[str, Any]):
    """CloudZero Billing Format (CBF) record structure.
    
    This class represents a CBF record that is created from LiteLLM usage data 
    for CloudZero integration. Since CBF field names contain forward slashes 
    (e.g., 'time/usage_start', 'cost/cost'), we use a Dict base class rather 
    than TypedDict to accommodate the special characters in field names.
    
    Expected CBF fields (per LIT-1907):
    - time/usage_start: ISO-formatted UTC datetime (Optional[str])
    - cost/cost: Billed cost (float)
    - resource/id: Model name (str)
    - usage/amount: Numeric value of tokens consumed (int)
    - usage/units: Description of units, e.g., 'tokens' (str)
    - resource/service: Model group (str)
    - resource/account: api_key_alias|api_key_prefix (str)
    - resource/region: Maps to CZRN region, e.g., 'cross-region' (str)
    - resource/usage_family: Provider (str)
    - action/operation: Team ID (str)
    - lineitem/type: Standard usage line item, e.g., 'Usage' (str)
    - resource/tag:provider: CZRN provider component (str)
    - resource/tag:model: CZRN cloud-local-id component (model) (str)
    - resource/tag:organization_alias: Organization alias if available (Optional[str])
    - resource/tag:project_alias: Project alias if available (Optional[str])
    - resource/tag:user_alias: User alias if available (Optional[str])
    - resource/tag:{key}: Various resource tags for dimensions and metrics (Optional[str])
    """
    pass


# Type alias for better readability in function signatures
CBFRecordDict = Dict[str, Any] 