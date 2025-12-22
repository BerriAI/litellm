from abc import ABC, abstractmethod
from typing import Literal

from litellm.proxy._types import CallInfo


class BaseBudgetAlertType(ABC):
    """Base class for different budget alert types"""

    @abstractmethod
    def get_event_message(self) -> str:
        """Return the event message for this alert type"""
        pass

    @abstractmethod
    def get_id(self, user_info: CallInfo) -> str:
        """Return the ID to use for caching/tracking this alert"""
        pass


class ProxyBudgetAlert(BaseBudgetAlertType):
    def get_event_message(self) -> str:
        return "Proxy Budget: "

    def get_id(self, user_info: CallInfo) -> str:
        return "default_id"


class SoftBudgetAlert(BaseBudgetAlertType):
    def get_event_message(self) -> str:
        return "Soft Budget Crossed: "

    def get_id(self, user_info: CallInfo) -> str:
        return user_info.token or "default_id"


class UserBudgetAlert(BaseBudgetAlertType):
    def get_event_message(self) -> str:
        return "User Budget: "

    def get_id(self, user_info: CallInfo) -> str:
        return user_info.user_id or "default_id"


class TeamBudgetAlert(BaseBudgetAlertType):
    def get_event_message(self) -> str:
        return "Team Budget: "

    def get_id(self, user_info: CallInfo) -> str:
        return user_info.team_id or "default_id"


class OrganizationBudgetAlert(BaseBudgetAlertType):
    def get_event_message(self) -> str:
        return "Organization Budget: "

    def get_id(self, user_info: CallInfo) -> str:
        return user_info.organization_id or "default_id"


class TokenBudgetAlert(BaseBudgetAlertType):
    def get_event_message(self) -> str:
        return "Key Budget: "

    def get_id(self, user_info: CallInfo) -> str:
        return user_info.token or "default_id"


class ProjectedLimitExceededAlert(BaseBudgetAlertType):
    def get_event_message(self) -> str:
        return "Key Budget: Projected Limit Exceeded"

    def get_id(self, user_info: CallInfo) -> str:
        return user_info.token or "default_id"


def get_budget_alert_type(
    type: Literal[
        "token_budget",
        "user_budget",
        "soft_budget",
        "max_budget_alert",
        "team_budget",
        "organization_budget",
        "proxy_budget",
        "projected_limit_exceeded",
    ],
) -> BaseBudgetAlertType:
    """Factory function to get the appropriate budget alert type class"""

    alert_types = {
        "proxy_budget": ProxyBudgetAlert(),
        "soft_budget": SoftBudgetAlert(),
        "user_budget": UserBudgetAlert(),
        "max_budget_alert": TokenBudgetAlert(),
        "team_budget": TeamBudgetAlert(),
        "organization_budget": OrganizationBudgetAlert(),
        "token_budget": TokenBudgetAlert(),
        "projected_limit_exceeded": ProjectedLimitExceededAlert(),
    }

    if type in alert_types:
        return alert_types[type]
    else:
        return ProxyBudgetAlert()
