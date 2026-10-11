import enum
from typing import Final


class LitellmUserRoles(str, enum.Enum):
    """
    Admin Roles:
    PROXY_ADMIN: admin over the platform
    PROXY_ADMIN_VIEW_ONLY: can login, view all own keys, view all spend
    ORG_ADMIN: admin over a specific organization, can create teams, users only within their organization

    Internal User Roles:
    INTERNAL_USER: can login, view/create/delete their own keys, view their spend
    INTERNAL_USER_VIEW_ONLY: can login, view their own keys, view their own spend


    Team Roles:
    TEAM: used for JWT auth


    Customer Roles:
    CUSTOMER: External users -> these are customers

    """

    # Admin Roles
    PROXY_ADMIN = "proxy_admin"
    PROXY_ADMIN_VIEW_ONLY = "proxy_admin_viewer"

    # Organization admins
    ORG_ADMIN = "org_admin"

    # Internal User Roles
    INTERNAL_USER = "internal_user"
    INTERNAL_USER_VIEW_ONLY = "internal_user_viewer"

    # Team Roles
    TEAM = "team"

    # Customer Roles - External users of proxy
    CUSTOMER = "customer"

    def __str__(self):
        return str(self.value)

    def values(self) -> list[str]:
        return list(self.__annotations__.keys())

    @property
    def description(self) -> str:
        """
        Descriptions for the enum values
        """
        descriptions: Final = {
            "proxy_admin": "admin over litellm proxy, has all permissions",
            "proxy_admin_viewer": "view all keys, view all spend",
            "internal_user": "view/create/delete their own keys, view their own spend",
            "internal_user_viewer": "view their own keys, view their own spend",
            "team": "team scope used for JWT auth",
            "customer": "customer",
        }
        return descriptions.get(self.value, "")

    @property
    def ui_label(self) -> str:
        """
        UI labels for the enum values
        """
        ui_labels: Final = {
            "proxy_admin": "Admin (All Permissions)",
            "proxy_admin_viewer": "Admin (View Only)",
            "internal_user": "Internal User (Create/Delete/View)",
            "internal_user_viewer": "Internal User (View Only)",
            "team": "Team",
            "customer": "Customer",
        }
        return ui_labels.get(self.value, "")

    @property
    def is_internal_user_role(self) -> bool:
        """returns true if this role is an `internal_user` or `internal_user_viewer` role"""
        return self.value in [
            self.INTERNAL_USER,
            self.INTERNAL_USER_VIEW_ONLY,
        ]


class KeyManagementRoutes(str, enum.Enum):
    """
    Enum for key management routes
    """

    # write routes
    KEY_GENERATE = "/key/generate"
    KEY_UPDATE = "/key/update"
    KEY_DELETE = "/key/delete"
    KEY_REGENERATE = "/key/regenerate"
    KEY_GENERATE_SERVICE_ACCOUNT = "/key/service-account/generate"
    KEY_REGENERATE_WITH_PATH_PARAM = "/key/{key_id}/regenerate"
    KEY_BLOCK = "/key/block"
    KEY_UNBLOCK = "/key/unblock"
    KEY_BULK_UPDATE = "/key/bulk_update"
    TEAM_KEY_BULK_UPDATE = "/team/key/bulk_update"
    KEY_RESET_SPEND = "/key/{key_id}/reset_spend"

    # Field-level opt-in permission (not a real HTTP route). When present in a
    # team's `team_member_permissions`, non-admin members of that team may set
    # `access_group_ids` on keys they create/update. Default-deny.
    KEY_ACCESS_GROUP_ASSIGNMENT = "/key/access_group_assignment"
    AUTO_ROUTER_MANAGE = "/auto_router/manage"

    # info and health routes
    KEY_INFO = "/key/info"
    KEY_HEALTH = "/key/health"

    # list routes
    KEY_LIST = "/key/list"
    KEY_ALIASES = "/key/aliases"

    # team usage routes
    TEAM_DAILY_ACTIVITY = "/team/daily/activity"
    TEAM_DAILY_ACTIVITY_AGGREGATED = "/team/daily/activity/aggregated"

    # team spend-log viewing
    SPEND_LOGS = "/spend/logs"
    SPEND_LOGS_V2 = "/spend/logs/v2"
