# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Generic, TypeVar, Optional
from typing_extensions import override

from ._base_client import BasePage, PageInfo, BaseSyncPage, BaseAsyncPage

__all__ = ["SyncPage", "AsyncPage"]

_T = TypeVar("_T")


class SyncPage(BaseSyncPage[_T], BasePage[_T], Generic[_T]):
    data: List[_T]
    has_more: Optional[bool] = None
    first_id: Optional[str] = None
    last_id: Optional[str] = None

    @override
    def _get_page_items(self) -> List[_T]:
        data = self.data
        if not data:
            return []
        return data

    @override
    def has_next_page(self) -> bool:
        has_more = self.has_more
        if has_more is not None and has_more is False:
            return False

        return super().has_next_page()

    @override
    def next_page_info(self) -> Optional[PageInfo]:
        if self._options.params.get("before_id"):
            first_id = self.first_id
            if not first_id:
                return None

            return PageInfo(params={"before_id": first_id})

        last_id = self.last_id
        if not last_id:
            return None

        return PageInfo(params={"after_id": last_id})


class AsyncPage(BaseAsyncPage[_T], BasePage[_T], Generic[_T]):
    data: List[_T]
    has_more: Optional[bool] = None
    first_id: Optional[str] = None
    last_id: Optional[str] = None

    @override
    def _get_page_items(self) -> List[_T]:
        data = self.data
        if not data:
            return []
        return data

    @override
    def has_next_page(self) -> bool:
        has_more = self.has_more
        if has_more is not None and has_more is False:
            return False

        return super().has_next_page()

    @override
    def next_page_info(self) -> Optional[PageInfo]:
        if self._options.params.get("before_id"):
            first_id = self.first_id
            if not first_id:
                return None

            return PageInfo(params={"before_id": first_id})

        last_id = self.last_id
        if not last_id:
            return None

        return PageInfo(params={"after_id": last_id})
