from litellm.proxy.auth.core.core import (
    AnonymousCore,
    AuthCore,
    AuthorizeCore,
    AuthPlanner,
    CustomAuthCore,
    JwtCore,
    KeyCore,
    ModelBudgetCore,
    OAuth2Core,
)


class RustCore(AuthCore):
    def keys(self) -> KeyCore:
        raise NotImplementedError

    def model_budgets(self) -> ModelBudgetCore:
        raise NotImplementedError

    def plan(self) -> AuthPlanner:
        raise NotImplementedError

    def anonymous(self) -> AnonymousCore:
        raise NotImplementedError

    def jwt(self) -> JwtCore:
        raise NotImplementedError

    def oauth2(self) -> OAuth2Core:
        raise NotImplementedError

    def custom(self) -> CustomAuthCore:
        raise NotImplementedError

    def authorize(self) -> AuthorizeCore:
        raise NotImplementedError
