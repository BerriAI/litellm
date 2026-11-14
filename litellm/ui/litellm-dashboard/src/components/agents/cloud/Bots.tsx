import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { CloudAgent, useCloudBots } from "./hooks";
import { CloudTable } from "./Table";

export default function Bots({ accessToken, sourceId }: { accessToken: string; sourceId: string }) {
  const { t } = useTranslation();
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const bots = useCloudBots(accessToken, sourceId, search, page);
  const columns = [
    {
      key: "bot",
      heading: t("agents.external.bot"),
      cell: (row: CloudAgent) => (
        <code title={row.external_id} className="block max-w-48 truncate text-xs">
          {row.external_id}
        </code>
      ),
    },
    {
      key: "actor",
      heading: t("agents.external.actor"),
      cell: (row: CloudAgent) => (
        <span className="block max-w-36 truncate" title={row.last_actor_id ?? undefined}>
          {row.last_actor_id || "-"}
        </span>
      ),
    },
    {
      key: "model",
      heading: t("agents.external.model"),
      cell: (row: CloudAgent) => (
        <span className="block max-w-40 truncate" title={row.last_model ?? undefined}>
          {row.last_model || "-"}
        </span>
      ),
    },
    {
      key: "tools",
      heading: t("agents.external.tools"),
      cell: (row: CloudAgent) => (
        <div className="flex min-w-40 max-w-64 flex-wrap gap-1">
          {row.tools.slice(0, 3).map((tool) => (
            <Badge variant="secondary" key={tool} title={tool} className="max-w-32 truncate">
              {tool}
            </Badge>
          ))}
          {row.tools.length > 3 && (
            <Badge variant="outline" title={row.tools.slice(3).join(", ")}>
              +{row.tools.length - 3}
            </Badge>
          )}
          {row.tools.length === 0 && "-"}
        </div>
      ),
    },
    {
      key: "requests",
      heading: t("agents.external.requests"),
      cell: (row: CloudAgent) => row.request_count.toLocaleString(),
    },
    {
      key: "first",
      heading: t("agents.external.firstSeen"),
      cell: (row: CloudAgent) => (
        <span className="whitespace-nowrap">{new Date(row.first_seen_at).toLocaleString()}</span>
      ),
    },
    {
      key: "last",
      heading: t("agents.external.lastSeen"),
      cell: (row: CloudAgent) => (
        <span className="whitespace-nowrap">{new Date(row.last_seen_at).toLocaleString()}</span>
      ),
    },
  ];
  return (
    <div className="space-y-4">
      <Input
        aria-label={t("agents.external.search")}
        placeholder={t("agents.external.search")}
        value={search}
        className="max-w-sm"
        onChange={(event) => {
          setSearch(event.target.value);
          setPage(1);
        }}
      />
      {bots.error && (
        <p role="alert" className="text-sm text-destructive">
          {bots.error.message}
        </p>
      )}
      {bots.isLoading ? (
        <p role="status">{t("agents.external.loading")}</p>
      ) : (
        <CloudTable rows={bots.data?.data ?? []} columns={columns} rowKey={(row) => row.external_id} />
      )}
      <div className="flex flex-wrap items-center justify-between gap-3 text-sm text-muted-foreground">
        <span>{t("agents.external.page", { page, count: bots.data?.total ?? 0 })}</span>
        <div className="flex gap-2">
          <Button variant="outline" disabled={page === 1 || bots.isFetching} onClick={() => setPage(page - 1)}>
            {t("agents.external.previous")}
          </Button>
          <Button
            variant="outline"
            disabled={!bots.data || page * bots.data.page_size >= bots.data.total || bots.isFetching}
            onClick={() => setPage(page + 1)}
          >
            {t("agents.external.next")}
          </Button>
        </div>
      </div>
    </div>
  );
}
