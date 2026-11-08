import { useState } from "react";
import { useInfiniteQuery } from "@tanstack/react-query";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useInfiniteKeys, useKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import { useInfiniteTeams, useTeam } from "@/app/(dashboard)/hooks/teams/useTeams";
import { useInfiniteUsers, useUserLookup } from "@/app/(dashboard)/hooks/users/useUsers";
import { PaginatedSearchSelect } from "@/components/shared/PaginatedSearchSelect";
import type { SearchSelectOption } from "@/components/shared/SearchSelect";
import { Button } from "@/components/ui/button";
import { backgroundFetchClient } from "@/lib/http/api";
import { getProxyBaseUrl } from "@/components/networking";
import { SCOPE_LABELS, type AlertModelsPage, type RuleInput } from "./usageAlertApi";

type ResourceScope = Exclude<RuleInput["scope_type"], "deployment">;
type ResourceValue = { id: string | null; label: string };
interface ResourceProps {
  scope: ResourceScope;
  value: string | null;
  onChange: (value: ResourceValue) => void;
}
interface ResourceQuery {
  isLoading: boolean;
  isError: boolean;
  isFetchingNextPage: boolean;
  hasNextPage: boolean;
  fetchNextPage: () => Promise<unknown>;
  refetch: () => Promise<unknown>;
}

const PAGE_SIZE = 20;
const ALL_LABELS = {
  deployment: "Entire gateway",
  key: "All keys",
  team: "All teams",
  user: "All users",
  model: "All models",
};
export const allResourcesLabel = (scope: RuleInput["scope_type"]) => ALL_LABELS[scope];

const fallbackLabel = (scope: ResourceScope, id: string) =>
  scope === "model" ? id : `${SCOPE_LABELS[scope]} ${id.length > 24 ? `…${id.slice(-8)}` : id}`;

function ResourcePicker({
  scope,
  value,
  onChange,
  options,
  query,
  onSearch,
}: ResourceProps & { options: SearchSelectOption[]; query: ResourceQuery; onSearch: (value: string) => void }) {
  const unique = Array.from(new Map(options.map((option) => [option.value, option])).values());
  const current =
    value && !unique.some((option) => option.value === value) ? [{ value, label: fallbackLabel(scope, value) }] : [];
  const items = [
    { value: "all", label: allResourcesLabel(scope) },
    ...[...current, ...unique].map((option) => ({ ...option, value: `entity:${option.value}` })),
  ];
  const changeResource = (next: string | null) => {
    const id = next?.startsWith("entity:") ? next.slice(7) : null;
    const label = items.find((option) => option.value === next)?.label ?? allResourcesLabel(scope);
    onChange({ id, label });
  };
  return (
    <>
      <PaginatedSearchSelect
        inputId="rule-resource"
        options={items}
        value={value === null ? "all" : `entity:${value}`}
        onValueChange={changeResource}
        onSearchChange={onSearch}
        onLoadMore={() => void query.fetchNextPage()}
        hasNextPage={query.hasNextPage}
        isLoading={query.isLoading}
        isFetchingNextPage={query.isFetchingNextPage}
        placeholder={`Search ${SCOPE_LABELS[scope].toLowerCase()}s`}
        emptyText="No matching resources"
      />
      {query.isError && (
        <div role="alert" className="flex items-center gap-2 text-xs text-destructive">
          Could not load resources.
          <Button type="button" variant="link" size="sm" onClick={() => void query.refetch()}>
            Retry
          </Button>
        </div>
      )}
    </>
  );
}

function KeyResource(props: ResourceProps) {
  const [search, setSearch] = useState("");
  const query = useInfiniteKeys(PAGE_SIZE, { search: search || undefined });
  const keys = (query.data?.pages ?? []).flatMap((page) => page.keys);
  const loaded = keys.some((key) => key.token === props.value);
  const selected = useKeys(1, 1, { keyHash: props.value }, { enabled: Boolean(props.value) && !loaded });
  const options = [...(selected.data?.keys ?? []), ...keys]
    .filter((key) => safeKeyId(key.token))
    .map((key) => ({ value: key.token, label: key.key_alias || key.key_name || fallbackLabel("key", key.token) }));
  return <ResourcePicker {...props} options={options} query={query} onSearch={setSearch} />;
}

function safeKeyId(value: string): boolean {
  if (!value || value === "unattributed") return false;
  return !value.startsWith("sk-") && !value.startsWith("raw:sk-");
}

function TeamResource(props: ResourceProps) {
  const [search, setSearch] = useState("");
  const query = useInfiniteTeams(PAGE_SIZE, search || undefined);
  const teams = (query.data?.pages ?? []).flatMap((page) => page.teams);
  const loaded = teams.some((team) => team.team_id === props.value);
  const selected = useTeam(props.value && !loaded ? props.value : undefined);
  const options = [...(selected.data ? [selected.data] : []), ...teams]
    .filter((team) => team.team_id !== "litellm-dashboard")
    .map((team) => ({ value: team.team_id, label: team.team_alias || fallbackLabel("team", team.team_id) }));
  return <ResourcePicker {...props} options={options} query={query} onSearch={setSearch} />;
}

function UserResource(props: ResourceProps) {
  const [search, setSearch] = useState("");
  const query = useInfiniteUsers(PAGE_SIZE, search || undefined);
  const users = (query.data?.pages ?? []).flatMap((page) => page.users);
  const loaded = users.some((user) => user.user_id === props.value);
  const selected = useUserLookup(props.value && !loaded ? props.value : null);
  const options = [...(selected.data ? [selected.data] : []), ...users]
    .filter((user) => user.user_id !== "default_user_id")
    .map((user) => ({
      value: user.user_id,
      label: user.user_alias || user.user_email || fallbackLabel("user", user.user_id),
    }));
  return <ResourcePicker {...props} options={options} query={query} onSearch={setSearch} />;
}

function ModelResource(props: ResourceProps) {
  const { accessToken, userId, userRole } = useAuthorized();
  const [search, setSearch] = useState("");
  const queryOptions = {
    queryKey: ["usageAlertModels", getProxyBaseUrl(), userId, userRole, search],
    initialPageParam: "",
    queryFn: async ({ pageParam, signal }: { pageParam: string; signal: AbortSignal }) => {
      const { data } = await backgroundFetchClient.GET("/alerts/models", {
        params: { query: { search: search || undefined, cursor: pageParam || undefined, limit: PAGE_SIZE } },
        signal,
      });
      if (!data) throw new Error("Could not load models");
      return data;
    },
    getNextPageParam: (lastPage: AlertModelsPage) => lastPage.next_cursor ?? undefined,
    enabled: Boolean(accessToken),
  };
  const query = useInfiniteQuery(queryOptions);
  const options = (query.data?.pages ?? []).flatMap((page) =>
    page.data.map((model) => ({ value: model.scope_id, label: model.label })),
  );
  const window = query.data?.pages[0];
  return (
    <div className="space-y-1.5">
      <ResourcePicker {...props} options={options} query={query} onSearch={setSearch} />
      {window && (
        <p className="text-xs text-muted-foreground">
          Models recorded {window.start_date}–{window.end_date} UTC.
        </p>
      )}
    </div>
  );
}

export default function UsageRuleResourceSelect(props: ResourceProps) {
  switch (props.scope) {
    case "key":
      return <KeyResource {...props} />;
    case "team":
      return <TeamResource {...props} />;
    case "user":
      return <UserResource {...props} />;
    case "model":
      return <ModelResource {...props} />;
  }
}
