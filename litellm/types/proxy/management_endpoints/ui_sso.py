from typing import List, Optional, TypedDict


class MicrosoftGraphAPIUserGroupDirectoryObject(TypedDict, total=False):
    """Model for Microsoft Graph API directory object"""

    odata_type: Optional[str]
    id: Optional[str]
    deletedDateTime: Optional[str]
    description: Optional[str]
    displayName: Optional[str]
    roleTemplateId: Optional[str]


class MicrosoftGraphAPIUserGroupResponse(TypedDict, total=False):
    """Model for Microsoft Graph API user groups response"""

    odata_context: Optional[str]
    odata_nextLink: Optional[str]
    value: Optional[List[MicrosoftGraphAPIUserGroupDirectoryObject]]
