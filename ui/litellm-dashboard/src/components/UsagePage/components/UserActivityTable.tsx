import React from "react";
import { Trans, useTranslation } from "react-i18next";

import { formatCompact } from "@/app/(dashboard)/usage/_components/components/overview/overviewData";

import type { UserActivityRow } from "../dailyActivityApi";
import { formatUserSpend, userPrimaryLabel, userRowKey, userSecondaryLabel } from "./userActivityData";

const USER_ROW_GRID = "grid w-full grid-cols-[minmax(0,1fr)_5.5rem_4.5rem_5rem_4.5rem_5.5rem] items-center gap-3";

interface UserActivityTableProps {
  rows: readonly UserActivityRow[];
  total: number;
  hasMore: boolean;
  loading: boolean;
  loadingMore: boolean;
  failed: boolean;
  onLoadMore: () => void;
  onRetryFirstPage: () => void;
  onRetryNextPage: () => void;
}

export const UserCell: React.FC<{ row: UserActivityRow }> = ({ row }) => {
  const secondary = userSecondaryLabel(row);
  return (
    <div className="min-w-0">
      <div className="truncate text-sm text-foreground">{userPrimaryLabel(row)}</div>
      {secondary && <div className="truncate text-xs text-muted-foreground">{secondary}</div>}
    </div>
  );
};

const UserActivityTable: React.FC<UserActivityTableProps> = ({
  rows,
  total,
  hasMore,
  loading,
  loadingMore,
  failed,
  onLoadMore,
  onRetryFirstPage,
  onRetryNextPage,
}) => {
  const { t } = useTranslation();
  const firstPageFailed = failed && rows.length === 0;
  const body = (() => {
    if (firstPageFailed) {
      return (
        <p className="px-4 py-8 text-center text-sm text-muted-foreground">
          {t("usage.userActivity.firstPageFailed")}{" "}
          <button type="button" className="font-medium text-foreground underline" onClick={onRetryFirstPage}>
            {t("usage.userActivity.retry")}
          </button>
        </p>
      );
    }
    if (loading) {
      return <p className="px-4 py-8 text-center text-sm text-muted-foreground">{t("usage.userActivity.loading")}</p>;
    }
    if (rows.length === 0) {
      return <p className="px-4 py-8 text-center text-sm text-muted-foreground">{t("usage.userActivity.empty")}</p>;
    }
    return (
      <div className="overflow-x-auto">
        <div className="min-w-[40rem]">
          <div
            aria-hidden="true"
            className={`${USER_ROW_GRID} border-y bg-muted/40 py-2 pl-10 pr-4 text-xs font-medium text-muted-foreground`}
          >
            <span>{t("usage.userActivity.columns.user")}</span>
            <span className="text-right">{t("usage.userActivity.columns.spend")}</span>
            <span className="text-right">{t("usage.userActivity.columns.requests")}</span>
            <span className="text-right">{t("usage.userActivity.columns.successful")}</span>
            <span className="text-right">{t("usage.userActivity.columns.failed")}</span>
            <span className="text-right">{t("usage.userActivity.columns.totalTokens")}</span>
          </div>
          <ul>
            {rows.map((row) => (
              <li key={userRowKey(row)} className={`${USER_ROW_GRID} border-b py-2.5 pl-10 pr-4 last:border-b-0`}>
                <UserCell row={row} />
                <span className="text-right text-sm tabular-nums text-foreground">{formatUserSpend(row.spend)}</span>
                <span className="text-right text-sm tabular-nums text-foreground">
                  {row.api_requests.toLocaleString()}
                </span>
                <span className="text-right text-sm tabular-nums text-foreground">
                  {row.successful_requests.toLocaleString()}
                </span>
                <span
                  className={`text-right text-sm tabular-nums ${row.failed_requests > 0 ? "text-destructive" : "text-muted-foreground"}`}
                >
                  {row.failed_requests.toLocaleString()}
                </span>
                <span className="text-right text-sm tabular-nums text-muted-foreground">
                  {formatCompact(row.total_tokens)}
                </span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    );
  })();

  const nextPageFailed = failed && !firstPageFailed;
  const showFooter = hasMore || loadingMore || nextPageFailed;
  const canLoadMore = hasMore && !loadingMore && !failed;

  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-xs text-muted-foreground tabular-nums" aria-live="polite">
          <Trans
            i18nKey="usage.userActivity.userCount"
            count={total}
            components={{ value: <>{total.toLocaleString()}</> }}
          />
        </span>
      </div>
      <div>{body}</div>
      {showFooter && (
        <div className="border-t px-4 py-2.5 text-xs text-muted-foreground">
          {loadingMore && <p>{t("usage.userActivity.loadingMore")}</p>}
          {failed && !firstPageFailed && (
            <p>
              {t("usage.userActivity.loadMoreFailed")}{" "}
              <button type="button" className="font-medium text-foreground underline" onClick={onRetryNextPage}>
                {t("usage.userActivity.retry")}
              </button>
            </p>
          )}
          {canLoadMore && (
            <button type="button" className="font-medium text-foreground underline" onClick={onLoadMore}>
              {t("usage.userActivity.loadMore")}
            </button>
          )}
        </div>
      )}
    </div>
  );
};

export default UserActivityTable;
