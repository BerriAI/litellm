"use client";

import { useLanguage } from "@/contexts/LanguageContext";
import { useMemo, useState } from "react";

import { useInfiniteSpendLogEndUsers } from "@/app/(dashboard)/hooks/spendLogs/useSpendLogEndUsers";
import { useInfiniteSpendLogUsers } from "@/app/(dashboard)/hooks/spendLogs/useSpendLogUsers";
import { useInfiniteKeyAliases } from "@/app/(dashboard)/hooks/keys/useKeyAliases";
import { useInfiniteModelInfo } from "@/app/(dashboard)/hooks/models/useModels";
import { DataTableFilterField } from "@/components/shared/DataTable";
import { PaginatedSearchSelect } from "@/components/shared/PaginatedSearchSelect";
import { SearchSelect, type SearchSelectOption } from "@/components/shared/SearchSelect";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

import type { Team } from "../../key_team_helpers/key_list";
import { CREDENTIAL_LABELS, ERROR_CODE_OPTIONS } from "../constants";
import { LOG_FILTER_IDS, type LogsWindow } from "./useLogFilterLogic";

const ALL_VALUE = "all";

const STATUS_FILTER_ITEMS = [
  { value: ALL_VALUE, label: "All Statuses" },
  { value: "success", label: "Success" },
  { value: "failure", label: "Failure" },
] as const;

const CACHE_FILTER_ITEMS = [
  { value: ALL_VALUE, label: "All Requests" },
  { value: "hit", label: "Cache Hit" },
  { value: "miss", label: "Cache Miss" },
] as const;

const CREDENTIAL_FILTER_ITEMS = [
  { value: ALL_VALUE, label: "All Credentials" },
  ...Object.entries(CREDENTIAL_LABELS).map(([value, label]) => ({ value, label })),
] as const;

const SPAN_TYPE_FILTER_ITEMS = [
  { value: ALL_VALUE, label: "All Types" },
  { value: "llm", label: "LLM" },
  { value: "agent", label: "Agent" },
  { value: "mcp", label: "MCP" },
  { value: "batch", label: "Batch" },
] as const;
const PAGE_SIZE = 50;

const SEARCH_INPUT_REASONS: ReadonlySet<string> = new Set(["input-change", "input-clear", "clear-press"]);

const asString = (value: unknown): string => (typeof value === "string" ? value : "");
const emptyToUndefined = (value: string): string | undefined => (value === "" ? undefined : value);

function TeamFilterField({
  value,
  onChange,
  teams,
}: {
  value: string;
  onChange: (value: string | undefined) => void;
  teams: Team[];
}) {
  const { t } = useLanguage();
  const options = useMemo<SearchSelectOption[]>(
    () =>
      teams.map((team) => ({
        label: team.team_alias || team.team_id,
        value: team.team_id,
        sublabel: team.team_id,
      })),
    [teams],
  );

  return (
    <DataTableFilterField label={t("Team ID")}>
      <SearchSelect
        options={options}
        value={value}
        onValueChange={(next) => onChange(next ?? undefined)}
        placeholder={t("Search or select a team")}
        emptyText={t("No teams found")}
      />
    </DataTableFilterField>
  );
}

function KeyAliasFilterField({
  value,
  onChange,
  teamId,
}: {
  value: string;
  onChange: (value: string | undefined) => void;
  teamId: string;
}) {
  const { t } = useLanguage();
  const [search, setSearch] = useState("");
  const { data, fetchNextPage, hasNextPage, isFetchingNextPage, isLoading } = useInfiniteKeyAliases(
    PAGE_SIZE,
    emptyToUndefined(search),
    emptyToUndefined(teamId),
  );

  const options = useMemo<SearchSelectOption[]>(() => {
    const seen = new Set<string>();
    return (data?.pages ?? []).flatMap((page) =>
      page.aliases.flatMap((alias) => {
        if (!alias || seen.has(alias)) return [];
        seen.add(alias);
        return [{ label: alias, value: alias }];
      }),
    );
  }, [data]);

  return (
    <DataTableFilterField label={t("Key Alias")}>
      <PaginatedSearchSelect
        options={options}
        value={value}
        onValueChange={(next) => onChange(next ?? undefined)}
        onSearchChange={setSearch}
        onLoadMore={() => void fetchNextPage()}
        hasNextPage={hasNextPage}
        isLoading={isLoading}
        isFetchingNextPage={isFetchingNextPage}
        placeholder={t("Search a key alias")}
        emptyText={t("No key aliases found")}
      />
    </DataTableFilterField>
  );
}

function ModelFilterField({ value, onChange }: { value: string; onChange: (value: string | undefined) => void }) {
  const { t } = useLanguage();
  const [search, setSearch] = useState("");
  const { data, fetchNextPage, hasNextPage, isFetchingNextPage, isLoading } = useInfiniteModelInfo(
    PAGE_SIZE,
    emptyToUndefined(search),
  );

  const options = useMemo<SearchSelectOption[]>(() => {
    const seen = new Set<string>();
    return (data?.pages ?? []).flatMap((page) =>
      page.data.flatMap((model) => {
        const modelId = model.model_info?.id ?? "";
        const modelName = model.model_name ?? "";
        if (!modelId || seen.has(modelId)) return [];
        seen.add(modelId);
        return [{ label: modelName || modelId, value: modelId, sublabel: t("Model ID: {id}", { id: modelId }) }];
      }),
    );
  }, [data, t]);

  return (
    <DataTableFilterField label={t("Model")}>
      <PaginatedSearchSelect
        options={options}
        value={value}
        onValueChange={(next) => onChange(next ?? undefined)}
        onSearchChange={setSearch}
        onLoadMore={() => void fetchNextPage()}
        hasNextPage={hasNextPage}
        isLoading={isLoading}
        isFetchingNextPage={isFetchingNextPage}
        placeholder={t("Search a model")}
        emptyText={t("No models found")}
      />
    </DataTableFilterField>
  );
}

function UserIdFilterField({
  value,
  onChange,
  logsWindow,
}: {
  value: string;
  onChange: (value: string | undefined) => void;
  logsWindow: LogsWindow;
}) {
  const { t } = useLanguage();
  const [search, setSearch] = useState("");
  const { data, fetchNextPage, hasNextPage, isFetchingNextPage, isLoading } = useInfiniteSpendLogUsers(
    logsWindow,
    PAGE_SIZE,
    emptyToUndefined(search),
  );

  const options = useMemo<SearchSelectOption[]>(() => {
    const seen = new Set<string>();
    return (data?.pages ?? []).flatMap((page) =>
      page.data.flatMap((userId) => {
        if (!userId || seen.has(userId)) return [];
        seen.add(userId);
        return [{ label: userId, value: userId }];
      }),
    );
  }, [data]);

  return (
    <DataTableFilterField label={t("User ID")}>
      <PaginatedSearchSelect
        options={options}
        value={value}
        onValueChange={(next) => onChange(next ?? undefined)}
        onSearchChange={setSearch}
        onLoadMore={() => void fetchNextPage()}
        hasNextPage={hasNextPage}
        isLoading={isLoading}
        isFetchingNextPage={isFetchingNextPage}
        placeholder={t("Search an internal user")}
        emptyText={t("No users found")}
      />
    </DataTableFilterField>
  );
}

function EndUserFilterField({
  value,
  onChange,
  logsWindow,
}: {
  value: string;
  onChange: (value: string | undefined) => void;
  logsWindow: LogsWindow;
}) {
  const { t } = useLanguage();
  const [search, setSearch] = useState("");
  const { data, fetchNextPage, hasNextPage, isFetchingNextPage, isLoading } = useInfiniteSpendLogEndUsers(
    logsWindow,
    PAGE_SIZE,
    emptyToUndefined(search),
  );

  const options = useMemo<SearchSelectOption[]>(() => {
    const seen = new Set<string>();
    return (data?.pages ?? []).flatMap((page) =>
      page.data.flatMap((endUser) => {
        if (!endUser || seen.has(endUser)) return [];
        seen.add(endUser);
        return [{ label: endUser, value: endUser }];
      }),
    );
  }, [data]);

  return (
    <DataTableFilterField label={t("End User")}>
      <PaginatedSearchSelect
        options={options}
        value={value}
        onValueChange={(next) => onChange(next ?? undefined)}
        onSearchChange={setSearch}
        onLoadMore={() => void fetchNextPage()}
        hasNextPage={hasNextPage}
        isLoading={isLoading}
        isFetchingNextPage={isFetchingNextPage}
        placeholder={t("Search an end user")}
        emptyText={t("No end users in this time range")}
      />
    </DataTableFilterField>
  );
}

function ErrorCodeFilterField({ value, onChange }: { value: string; onChange: (value: string | undefined) => void }) {
  const { t } = useLanguage();
  const [query, setQuery] = useState("");

  const options = useMemo<SearchSelectOption[]>(() => {
    const trimmed = query.trim();
    const lowered = trimmed.toLowerCase();
    const localized = ERROR_CODE_OPTIONS.map((option) => ({ ...option, label: t(option.label) }));
    const matches = localized.filter(
      (option) => option.value.includes(trimmed) || option.label.toLowerCase().includes(lowered),
    );
    const isKnownCode = localized.some((option) => option.value === trimmed || option.label.toLowerCase() === lowered);
    if (trimmed === "" || isKnownCode) return matches;
    return [...matches, { label: t("Use custom code: {code}", { code: trimmed }), value: trimmed }];
  }, [query, t]);

  const selected = useMemo<SearchSelectOption | null>(() => {
    if (value === "") return null;
    const known = ERROR_CODE_OPTIONS.find((option) => option.value === value);
    return known ? { ...known, label: t(known.label) } : { label: value, value };
  }, [value, t]);

  const items = useMemo<SearchSelectOption[]>(() => {
    if (selected === null) return options;
    if (options.some((option) => option.value === selected.value)) return options;
    return [selected, ...options];
  }, [options, selected]);

  return (
    <DataTableFilterField label={t("Error Code")}>
      <Combobox
        items={items}
        value={selected}
        onValueChange={(item: SearchSelectOption | null) => onChange(emptyToUndefined(item?.value ?? ""))}
        onInputValueChange={(next, eventDetails) => setQuery(SEARCH_INPUT_REASONS.has(eventDetails.reason) ? next : "")}
        onOpenChange={(nextOpen) => {
          if (!nextOpen) setQuery("");
        }}
        isItemEqualToValue={(a: SearchSelectOption, b: SearchSelectOption) => a.value === b.value}
        itemToStringLabel={(item: SearchSelectOption) => item.label}
        filter={null}
      >
        <ComboboxInput
          onFocus={(event) => event.currentTarget.select()}
          placeholder={t("Select or type an error code")}
          showClear={value !== ""}
          className="w-full"
        />
        <ComboboxContent>
          <ComboboxEmpty>{t("No error codes found")}</ComboboxEmpty>
          <ComboboxList data-testid="error-code-filter-list">
            {(item: SearchSelectOption) => (
              <ComboboxItem key={item.value} value={item}>
                {item.label}
              </ComboboxItem>
            )}
          </ComboboxList>
        </ComboboxContent>
      </Combobox>
    </DataTableFilterField>
  );
}

interface RequestLogsFiltersProps {
  get: (columnId: string) => unknown;
  set: (columnId: string, value: unknown) => void;
  teams: Team[];
  logsWindow: LogsWindow;
}

export function RequestLogsFilters({ get, set, teams, logsWindow }: RequestLogsFiltersProps) {
  const { t } = useLanguage();
  const valueOf = (id: string): string => asString(get(id));
  const setter = (id: string) => (next: string | undefined) => set(id, next);
  // 只翻译固定筛选项，密钥别名、用户名称和模型名称仍使用服务端原值。
  const spanItems = SPAN_TYPE_FILTER_ITEMS.map((item) => ({ ...item, label: t(item.label) }));
  const statusItems = STATUS_FILTER_ITEMS.map((item) => ({ ...item, label: t(item.label) }));
  const cacheItems = CACHE_FILTER_ITEMS.map((item) => ({ ...item, label: t(item.label) }));
  const credentialItems = CREDENTIAL_FILTER_ITEMS.map((item) => ({ ...item, label: t(item.label) }));

  return (
    <>
      <TeamFilterField
        value={valueOf(LOG_FILTER_IDS.TEAM_ID)}
        onChange={setter(LOG_FILTER_IDS.TEAM_ID)}
        teams={teams}
      />

      <DataTableFilterField label={t("Span Type")}>
        <Select
          items={spanItems}
          value={valueOf(LOG_FILTER_IDS.SPAN_TYPE) === "" ? ALL_VALUE : valueOf(LOG_FILTER_IDS.SPAN_TYPE)}
          onValueChange={(next) =>
            set(LOG_FILTER_IDS.SPAN_TYPE, next === null || next === ALL_VALUE ? undefined : next)
          }
        >
          <SelectTrigger className="w-full">
            <SelectValue placeholder={t("All Types")} />
          </SelectTrigger>
          <SelectContent>
            {spanItems.map((item) => (
              <SelectItem key={item.value} value={item.value}>
                {item.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </DataTableFilterField>

      <DataTableFilterField label={t("Status")}>
        <Select
          items={statusItems}
          value={valueOf(LOG_FILTER_IDS.STATUS) === "" ? ALL_VALUE : valueOf(LOG_FILTER_IDS.STATUS)}
          onValueChange={(next) => set(LOG_FILTER_IDS.STATUS, next === null || next === ALL_VALUE ? undefined : next)}
        >
          <SelectTrigger className="w-full">
            <SelectValue placeholder={t("All Statuses")} />
          </SelectTrigger>
          <SelectContent>
            {statusItems.map((item) => (
              <SelectItem key={item.value} value={item.value}>
                {item.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </DataTableFilterField>

      <DataTableFilterField label={t("Cache")}>
        <Select
          items={cacheItems}
          value={valueOf(LOG_FILTER_IDS.CACHE_STATUS) === "" ? ALL_VALUE : valueOf(LOG_FILTER_IDS.CACHE_STATUS)}
          onValueChange={(next) =>
            set(LOG_FILTER_IDS.CACHE_STATUS, next === null || next === ALL_VALUE ? undefined : next)
          }
        >
          <SelectTrigger className="w-full">
            <SelectValue placeholder={t("All Requests")} />
          </SelectTrigger>
          <SelectContent>
            {cacheItems.map((item) => (
              <SelectItem key={item.value} value={item.value}>
                {item.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </DataTableFilterField>

      <DataTableFilterField label={t("Credential")}>
        <Select
          items={credentialItems}
          value={valueOf(LOG_FILTER_IDS.CREDENTIAL) === "" ? ALL_VALUE : valueOf(LOG_FILTER_IDS.CREDENTIAL)}
          onValueChange={(next) =>
            set(LOG_FILTER_IDS.CREDENTIAL, next === null || next === ALL_VALUE ? undefined : next)
          }
        >
          <SelectTrigger className="w-full">
            <SelectValue placeholder={t("All Credentials")} />
          </SelectTrigger>
          <SelectContent>
            {credentialItems.map((item) => (
              <SelectItem key={item.value} value={item.value}>
                {item.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </DataTableFilterField>

      <KeyAliasFilterField
        value={valueOf(LOG_FILTER_IDS.KEY_ALIAS)}
        onChange={setter(LOG_FILTER_IDS.KEY_ALIAS)}
        teamId={valueOf(LOG_FILTER_IDS.TEAM_ID)}
      />

      <UserIdFilterField
        value={valueOf(LOG_FILTER_IDS.USER_ID)}
        onChange={setter(LOG_FILTER_IDS.USER_ID)}
        logsWindow={logsWindow}
      />

      <EndUserFilterField
        value={valueOf(LOG_FILTER_IDS.END_USER)}
        onChange={setter(LOG_FILTER_IDS.END_USER)}
        logsWindow={logsWindow}
      />

      <ErrorCodeFilterField value={valueOf(LOG_FILTER_IDS.ERROR_CODE)} onChange={setter(LOG_FILTER_IDS.ERROR_CODE)} />

      <DataTableFilterField label={t("Error Message")}>
        <Input
          value={valueOf(LOG_FILTER_IDS.ERROR_MESSAGE)}
          onChange={(event) => set(LOG_FILTER_IDS.ERROR_MESSAGE, emptyToUndefined(event.target.value))}
          placeholder={t("Enter error message…")}
        />
      </DataTableFilterField>

      <DataTableFilterField label={t("Key Hash")}>
        <Input
          value={valueOf(LOG_FILTER_IDS.KEY_HASH)}
          onChange={(event) => set(LOG_FILTER_IDS.KEY_HASH, emptyToUndefined(event.target.value))}
          placeholder={t("Enter key hash…")}
        />
      </DataTableFilterField>

      <DataTableFilterField label={t("Session ID")}>
        <Input
          value={valueOf(LOG_FILTER_IDS.SESSION_ID)}
          onChange={(event) => set(LOG_FILTER_IDS.SESSION_ID, emptyToUndefined(event.target.value))}
          placeholder={t("Enter session ID…")}
        />
      </DataTableFilterField>

      <ModelFilterField value={valueOf(LOG_FILTER_IDS.MODEL_ID)} onChange={setter(LOG_FILTER_IDS.MODEL_ID)} />

      <DataTableFilterField label={t("Public model / search tool")}>
        <Input
          value={valueOf(LOG_FILTER_IDS.PUBLIC_MODEL_OR_SEARCH_TOOL)}
          onChange={(event) => set(LOG_FILTER_IDS.PUBLIC_MODEL_OR_SEARCH_TOOL, emptyToUndefined(event.target.value))}
          placeholder={t("Enter public model or search tool…")}
        />
      </DataTableFilterField>
    </>
  );
}
