import React, { useEffect, useLayoutEffect, useRef, useState } from "react";

import type { FetchUserPage } from "./userActivityData";
import { formatUserSpend, userPrimaryLabel, userRowKey, userSecondaryLabel } from "./userActivityData";
import type { UserActivityRow } from "../dailyActivityApi";

const TOP_USERS_LIMIT = 10;
const TOP_USER_ROW_GRID = "grid w-full grid-cols-[minmax(0,1fr)_5.5rem_4.5rem] items-center gap-4";

interface TopUsersViewProps {
  fetchUserPage: FetchUserPage;
}

const TopUsersView: React.FC<TopUsersViewProps> = ({ fetchUserPage }) => {
  const [state, setState] = useState<{
    scope: FetchUserPage;
    rows: UserActivityRow[];
    failed: boolean;
  } | null>(null);
  const [retryToken, setRetryToken] = useState(0);
  const requestIdRef = useRef(0);
  const activeFetcherRef = useRef(fetchUserPage);

  useLayoutEffect(() => {
    activeFetcherRef.current = fetchUserPage;
  }, [fetchUserPage]);

  useEffect(() => {
    const requestId = ++requestIdRef.current;
    let cancelled = false;
    void fetchUserPage(0, TOP_USERS_LIMIT)
      .then((page) => {
        if (cancelled || requestIdRef.current !== requestId || activeFetcherRef.current !== fetchUserPage) return;
        setState({ scope: fetchUserPage, rows: page.users, failed: false });
      })
      .catch((error: unknown) => {
        if (cancelled || requestIdRef.current !== requestId || activeFetcherRef.current !== fetchUserPage) return;
        console.error("Top users request failed:", error);
        setState({ scope: fetchUserPage, rows: [], failed: true });
      });
    return () => {
      cancelled = true;
    };
  }, [fetchUserPage, retryToken]);

  const current = state?.scope === fetchUserPage ? state : null;
  const loading = current === null;

  if (current?.failed) {
    return (
      <p className="px-1 py-6 text-center text-sm text-muted-foreground">
        Could not load top users.{" "}
        <button
          type="button"
          className="font-medium text-foreground underline"
          onClick={() => setRetryToken((token) => token + 1)}
        >
          Retry
        </button>
      </p>
    );
  }
  if (loading) {
    return <p className="px-1 py-6 text-center text-sm text-muted-foreground">Loading users...</p>;
  }
  if (current.rows.length === 0) {
    return <p className="px-1 py-6 text-center text-sm text-muted-foreground">No user activity in this date range</p>;
  }
  const maxSpend = Math.max(0, ...current.rows.map((row) => row.spend));
  return (
    <ul>
      <li
        aria-hidden="true"
        className={`${TOP_USER_ROW_GRID} border-b pb-1.5 text-xs font-medium text-muted-foreground`}
      >
        <span>User</span>
        <span className="text-right">Spend</span>
        <span className="text-right">Requests</span>
      </li>
      {current.rows.map((row) => {
        const secondary = userSecondaryLabel(row);
        const share = maxSpend > 0 ? (row.spend / maxSpend) * 100 : 0;
        return (
          <li key={userRowKey(row)} className={`${TOP_USER_ROW_GRID} border-b py-2.5 last:border-b-0`}>
            <div className="min-w-0">
              <div className="truncate text-sm text-foreground">{userPrimaryLabel(row)}</div>
              {secondary && <div className="truncate text-xs text-muted-foreground">{secondary}</div>}
              <div aria-hidden="true" className="mt-1 h-1 w-full max-w-xs overflow-hidden rounded-full bg-muted">
                <div
                  className="h-full rounded-full opacity-80"
                  style={{ width: `${share}%`, backgroundColor: "#2b3fd6" }}
                />
              </div>
            </div>
            <span className="text-right text-sm tabular-nums text-foreground">{formatUserSpend(row.spend)}</span>
            <span className="text-right text-sm tabular-nums text-muted-foreground">
              {row.api_requests.toLocaleString()}
            </span>
          </li>
        );
      })}
    </ul>
  );
};

export default TopUsersView;
