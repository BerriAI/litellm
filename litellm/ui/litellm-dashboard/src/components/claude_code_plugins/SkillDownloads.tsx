import { useInfiniteQuery, useMutation, type QueryFunctionContext } from "@tanstack/react-query";
import { downloadBlob } from "@/components/EntityUsageExport/utils";
import { RefreshCw } from "lucide-react";
import { useTranslation } from "react-i18next";
import { apiClient } from "@/components/networking";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { Plugin, PluginDownloadsResponse } from "./types";

export const skillDownloadsQueryKey = (skill: Pick<Plugin, "id" | "name">, accessToken?: string | null) =>
  ["skill-downloads", skill.id, skill.name, accessToken] as const;

interface SkillDownloadsProps {
  skill: Pick<Plugin, "id" | "name">;
  accessToken?: string | null;
}

export function SkillDownloads({ skill, accessToken }: SkillDownloadsProps) {
  const { t, i18n } = useTranslation();
  const exportCsv = useMutation({
    mutationFn: async () => {
      const blob = await apiClient.getBlob(`/skill-hub/skills/${encodeURIComponent(skill.name)}/downloads/export`, {
        accessToken,
      });
      downloadBlob(blob, `${skill.name}.csv`);
    },
  });
  const queryOptions = {
    queryKey: skillDownloadsQueryKey(skill, accessToken),
    queryFn: ({ pageParam, signal }: QueryFunctionContext<ReturnType<typeof skillDownloadsQueryKey>, string | null>) =>
      apiClient.get<PluginDownloadsResponse>(`/skill-hub/skills/${encodeURIComponent(skill.name)}/downloads`, {
        accessToken,
        query: { limit: 25, cursor: pageParam },
        signal,
      }),
    initialPageParam: null as string | null,
    getNextPageParam: (page: PluginDownloadsResponse) => page.next_cursor,
    staleTime: 0,
    refetchOnMount: true,
    retry: false,
  };
  const { data, isPending, isError, isFetching, isFetchingNextPage, hasNextPage, fetchNextPage, refetch } =
    useInfiniteQuery(queryOptions);
  const downloads = data?.pages.flatMap((page) => page.downloads) ?? [];
  const isEmpty = data && !isError && downloads.length === 0;
  const unavailable = t("claudeCodePlugins.skillDetail.downloads.unavailable");

  return (
    <section aria-label={t("claudeCodePlugins.skillDetail.downloads.title")} className="space-y-4">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-normal">{t("claudeCodePlugins.skillDetail.downloads.title")}</h2>
          <p className="mt-1 max-w-2xl text-sm text-muted-foreground">
            {t("claudeCodePlugins.skillDetail.downloads.description")}
          </p>
          <p className="mt-1 text-sm text-muted-foreground">
            {t("claudeCodePlugins.skillDetail.downloads.attributionNote")}
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" disabled={exportCsv.isPending} onClick={() => exportCsv.mutate()}>
            {t(
              exportCsv.isPending
                ? "claudeCodePlugins.skillDetail.downloads.exporting"
                : "claudeCodePlugins.skillDetail.downloads.downloadCsv",
            )}
          </Button>
          <Button variant="outline" size="sm" disabled={isFetching} onClick={() => void refetch()}>
            <RefreshCw className="size-4" />
            {t("claudeCodePlugins.skillDetail.downloads.refresh")}
          </Button>
        </div>
      </div>
      {exportCsv.isError && (
        <p role="alert" className="text-sm text-destructive">
          {t("claudeCodePlugins.skillDetail.downloads.exportFailed")}
        </p>
      )}
      {data && (
        <dl className="flex flex-wrap gap-x-8 gap-y-4">
          <div>
            <dt className="text-sm text-muted-foreground">{t("claudeCodePlugins.skillDetail.downloads.total")}</dt>
            <dd aria-label={t("claudeCodePlugins.skillDetail.downloads.total")} className="text-3xl tabular-nums">
              {data.pages[0].total_downloads.toLocaleString(i18n.language)}
            </dd>
          </div>
          <div>
            <dt className="text-sm text-muted-foreground">
              {t("claudeCodePlugins.skillDetail.downloads.anonymousTotal")}
            </dt>
            <dd
              aria-label={t("claudeCodePlugins.skillDetail.downloads.anonymousTotal")}
              className="text-3xl tabular-nums"
            >
              {data.pages[0].anonymous_downloads.toLocaleString(i18n.language)}
            </dd>
          </div>
        </dl>
      )}
      {isPending && <p role="status">{t("claudeCodePlugins.skillDetail.downloads.loading")}</p>}
      {isError && (
        <p role="alert" className="text-sm text-destructive">
          {t("claudeCodePlugins.skillDetail.downloads.loadFailed")}
        </p>
      )}
      {isFetching && !isPending && (
        <p role="status" className="text-sm text-muted-foreground">
          {t(
            isFetchingNextPage
              ? "claudeCodePlugins.skillDetail.downloads.loadingMore"
              : "claudeCodePlugins.skillDetail.downloads.refreshing",
          )}
        </p>
      )}
      {isEmpty && <p className="text-sm text-muted-foreground">{t("claudeCodePlugins.skillDetail.downloads.empty")}</p>}
      {downloads.length > 0 && (
        <Table aria-label={t("claudeCodePlugins.skillDetail.downloads.title")}>
          <TableHeader>
            <TableRow>
              <TableHead>{t("claudeCodePlugins.skillDetail.downloads.downloadedAt")}</TableHead>
              <TableHead>{t("claudeCodePlugins.skillDetail.downloads.downloader")}</TableHead>
              <TableHead>{t("claudeCodePlugins.skillDetail.downloads.key")}</TableHead>
              <TableHead>{t("claudeCodePlugins.skillDetail.downloads.team")}</TableHead>
              <TableHead>{t("claudeCodePlugins.skillDetail.downloads.version")}</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {downloads.map((event) => (
              <TableRow key={event.id}>
                <TableCell>
                  <time dateTime={event.downloaded_at}>
                    {new Date(event.downloaded_at).toLocaleString(i18n.language)}
                  </time>
                </TableCell>
                <TableCell>
                  {event.authenticated
                    ? event.user_email || event.user_id || t("claudeCodePlugins.skillDetail.downloads.unidentified")
                    : t("claudeCodePlugins.skillDetail.downloads.anonymous")}
                </TableCell>
                <TableCell>{event.key_alias || unavailable}</TableCell>
                <TableCell>{event.team_alias || event.team_id || unavailable}</TableCell>
                <TableCell>{event.version || unavailable}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
      {hasNextPage && (
        <Button variant="outline" disabled={isFetching} onClick={() => void fetchNextPage()}>
          {t("claudeCodePlugins.skillDetail.downloads.loadMore")}
        </Button>
      )}
    </section>
  );
}
