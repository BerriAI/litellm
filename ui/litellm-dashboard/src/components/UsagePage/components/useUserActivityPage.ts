import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";

import type { UserActivityRow } from "../dailyActivityApi";
import type { FetchUserPage } from "./userActivityData";
import { mergeUserActivityPages } from "./userActivityData";

export interface UserPageState {
  scope: FetchUserPage;
  rows: UserActivityRow[];
  total: number;
  nextOffset: number;
  hasMore: boolean;
  loading: boolean;
  loadingMore: boolean;
  failed: boolean;
}

export const emptyUserPageState = (scope: FetchUserPage, loading: boolean): UserPageState => ({
  scope,
  rows: [],
  total: 0,
  nextOffset: 0,
  hasMore: false,
  loading,
  loadingMore: false,
  failed: false,
});

export function useUserActivityPage(fetchUserPage: FetchUserPage, pageSize: number) {
  const [pageState, setPageState] = useState<UserPageState | null>(null);
  const [retryToken, setRetryToken] = useState(0);
  const requestIdRef = useRef(0);
  const loadingMoreRef = useRef(false);
  const activeFetcherRef = useRef(fetchUserPage);

  useLayoutEffect(() => {
    activeFetcherRef.current = fetchUserPage;
  }, [fetchUserPage]);

  useEffect(() => {
    const requestId = ++requestIdRef.current;
    loadingMoreRef.current = false;
    let cancelled = false;
    void fetchUserPage(0, pageSize)
      .then((page) => {
        if (cancelled || requestIdRef.current !== requestId || activeFetcherRef.current !== fetchUserPage) return;
        const nextPageState: UserPageState = {
          scope: fetchUserPage,
          rows: page.users,
          total: page.total_users,
          nextOffset: page.users.length,
          hasMore: page.users.length > 0 && page.users.length < page.total_users,
          loading: false,
          loadingMore: false,
          failed: false,
        };
        setPageState(nextPageState);
      })
      .catch((error: unknown) => {
        if (cancelled || requestIdRef.current !== requestId || activeFetcherRef.current !== fetchUserPage) return;
        console.error("User activity page request failed:", error);
        setPageState({ ...emptyUserPageState(fetchUserPage, false), failed: true });
      });
    return () => {
      cancelled = true;
    };
  }, [fetchUserPage, pageSize, retryToken]);

  const currentPageState = pageState?.scope === fetchUserPage ? pageState : null;

  const loadMore = useCallback(() => {
    if (currentPageState === null) return;
    if (currentPageState.loading || currentPageState.loadingMore || !currentPageState.hasMore) return;
    if (loadingMoreRef.current) return;
    const requestId = requestIdRef.current;
    const offset = currentPageState.nextOffset;
    loadingMoreRef.current = true;
    setPageState((current) => (current?.scope === fetchUserPage ? { ...current, loadingMore: true } : current));
    void fetchUserPage(offset, pageSize)
      .then((page) => {
        if (requestIdRef.current !== requestId || activeFetcherRef.current !== fetchUserPage) return;
        setPageState((current) => {
          if (current?.scope !== fetchUserPage) return current;
          const merged = mergeUserActivityPages(current.rows, page.users, page.total_users, current.nextOffset);
          return {
            ...current,
            rows: merged.rows,
            total: page.total_users,
            nextOffset: merged.nextOffset,
            hasMore: merged.hasMore,
            loadingMore: false,
            failed: false,
          };
        });
      })
      .catch((error: unknown) => {
        if (requestIdRef.current !== requestId || activeFetcherRef.current !== fetchUserPage) return;
        console.error("User activity page request failed:", error);
        setPageState((current) =>
          current?.scope === fetchUserPage ? { ...current, loadingMore: false, failed: true } : current,
        );
      })
      .finally(() => {
        if (requestIdRef.current === requestId) loadingMoreRef.current = false;
      });
  }, [currentPageState, fetchUserPage, pageSize]);

  const retryFirstPage = useCallback(() => {
    setPageState(emptyUserPageState(fetchUserPage, true));
    setRetryToken((token) => token + 1);
  }, [fetchUserPage]);

  const retryNextPage = useCallback(() => {
    setPageState((current) => (current?.scope === fetchUserPage ? { ...current, failed: false } : current));
    loadMore();
  }, [fetchUserPage, loadMore]);

  return {
    rows: currentPageState?.rows ?? [],
    total: currentPageState?.total ?? 0,
    hasMore: currentPageState?.hasMore ?? false,
    loading: currentPageState?.loading ?? true,
    loadingMore: currentPageState?.loadingMore ?? false,
    failed: currentPageState?.failed ?? false,
    loadMore,
    retryFirstPage,
    retryNextPage,
  };
}
