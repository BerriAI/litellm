"use client";

import type { ParseKeys } from "i18next";
import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";

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
const KEY_ALIAS_OPERATORS = [
  { value: "is", label: "is" },
  { value: "is not", label: "is not" },
] as const;
type KeyAliasOperator = (typeof KEY_ALIAS_OPERATORS)[number]["value"];

interface FilterItem {
  value: string;
  labelKey: ParseKeys;
}

const STATUS_FILTER_ITEMS: readonly FilterItem[] = [
  { value: ALL_VALUE, labelKey: "logs.request.filters.allStatuses" },
  { value: "success", labelKey: "logs.request.filters.statusOptions.success" },
  { value: "failure", labelKey: "logs.request.filters.statusOptions.failure" },
];

const CACHE_FILTER_ITEMS: readonly FilterItem[] = [
  { value: ALL_VALUE, labelKey: "logs.request.filters.allRequests" },
  { value: "hit", labelKey: "logs.request.filters.cacheOptions.hit" },
  { value: "miss", labelKey: "logs.request.filters.cacheOptions.miss" },
];

const CREDENTIAL_FILTER_ITEMS: readonly FilterItem[] = [
  { value: ALL_VALUE, labelKey: "logs.request.filters.allCredentials" },
  ...Object.entries(CREDENTIAL_LABELS).map(([value, labelKey]) => ({ value, labelKey })),
];

const SPAN_TYPE_FILTER_ITEMS: readonly FilterItem[] = [
  { value: ALL_VALUE, labelKey: "logs.request.filters.allTypes" },
  { value: "llm", labelKey: "logs.common.spanTypes.llm" },
  { value: "agent", labelKey: "logs.common.spanTypes.agent" },
  { value: "mcp", labelKey: "logs.common.spanTypes.mcp" },
  { value: "batch", labelKey: "logs.common.spanTypes.batch" },
];
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
  const { t } = useTranslation();
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
    <DataTableFilterField label={t("logs.request.filters.teamId")}>
      <SearchSelect
        options={options}
        value={value}
        onValueChange={(next) => onChange(next ?? undefined)}
        placeholder={t("logs.request.filters.teamPlaceholder")}
        emptyText={t("logs.request.filters.noTeams")}
      />
    </DataTableFilterField>
  );
}

function KeyAliasFilterField({
  value,
  excludeValue,
  onChange,
  onExcludeChange,
  teamId,
}: {
  value: string;
  excludeValue: string;
  onChange: (value: string | undefined) => void;
  onExcludeChange: (value: string | undefined) => void;
  teamId: string;
}) {
  const { t } = useTranslation();
  const [search, setSearch] = useState("");
  const [operator, setOperator] = useState<KeyAliasOperator>(excludeValue !== "" ? "is not" : "is");
  let activeOperator: KeyAliasOperator = operator;
  if (excludeValue !== "") {
    activeOperator = "is not";
  } else if (value !== "") {
    activeOperator = "is";
  }
  const handleOperatorChange = (next: KeyAliasOperator | null) => {
    if (next === null || next === activeOperator) return;
    const selectedAlias = value || excludeValue || undefined;
    setOperator(next);
    onChange(next === "is" ? selectedAlias : undefined);
    onExcludeChange(next === "is not" ? selectedAlias : undefined);
  };
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
    <DataTableFilterField label={t("logs.request.filters.keyAlias")}>
      <Select items={KEY_ALIAS_OPERATORS} value={activeOperator} onValueChange={handleOperatorChange}>
        <SelectTrigger className="w-full" aria-label="Key Alias operator">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {KEY_ALIAS_OPERATORS.map((item) => (
            <SelectItem key={item.value} value={item.value}>
              {item.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <PaginatedSearchSelect
        options={options}
        value={activeOperator === "is not" ? excludeValue : value}
        onValueChange={(next) =>
          activeOperator === "is not" ? onExcludeChange(next ?? undefined) : onChange(next ?? undefined)
        }
        onSearchChange={setSearch}
        onLoadMore={() => void fetchNextPage()}
        hasNextPage={hasNextPage}
        isLoading={isLoading}
        isFetchingNextPage={isFetchingNextPage}
        placeholder={t("logs.request.filters.keyAliasPlaceholder")}
        emptyText={t("logs.request.filters.noKeyAliases")}
      />
    </DataTableFilterField>
  );
}

function ModelFilterField({ value, onChange }: { value: string; onChange: (value: string | undefined) => void }) {
  const { t } = useTranslation();
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
        return [
          {
            label: modelName || modelId,
            value: modelId,
            sublabel: t("logs.request.filters.modelIdSublabel", { modelId }),
          },
        ];
      }),
    );
  }, [data, t]);

  return (
    <DataTableFilterField label={t("logs.request.filters.model")}>
      <PaginatedSearchSelect
        options={options}
        value={value}
        onValueChange={(next) => onChange(next ?? undefined)}
        onSearchChange={setSearch}
        onLoadMore={() => void fetchNextPage()}
        hasNextPage={hasNextPage}
        isLoading={isLoading}
        isFetchingNextPage={isFetchingNextPage}
        placeholder={t("logs.request.filters.modelPlaceholder")}
        emptyText={t("logs.request.filters.noModels")}
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
  const { t } = useTranslation();
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
    <DataTableFilterField label={t("logs.request.filters.userId")}>
      <PaginatedSearchSelect
        options={options}
        value={value}
        onValueChange={(next) => onChange(next ?? undefined)}
        onSearchChange={setSearch}
        onLoadMore={() => void fetchNextPage()}
        hasNextPage={hasNextPage}
        isLoading={isLoading}
        isFetchingNextPage={isFetchingNextPage}
        placeholder={t("logs.request.filters.userPlaceholder")}
        emptyText={t("logs.request.filters.noUsers")}
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
  const { t } = useTranslation();
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
    <DataTableFilterField label={t("logs.request.filters.endUser")}>
      <PaginatedSearchSelect
        options={options}
        value={value}
        onValueChange={(next) => onChange(next ?? undefined)}
        onSearchChange={setSearch}
        onLoadMore={() => void fetchNextPage()}
        hasNextPage={hasNextPage}
        isLoading={isLoading}
        isFetchingNextPage={isFetchingNextPage}
        placeholder={t("logs.request.filters.endUserPlaceholder")}
        emptyText={t("logs.request.filters.noEndUsers")}
      />
    </DataTableFilterField>
  );
}

function ErrorCodeFilterField({ value, onChange }: { value: string; onChange: (value: string | undefined) => void }) {
  const { t } = useTranslation();
  const [query, setQuery] = useState("");

  const errorCodeOptions = useMemo<SearchSelectOption[]>(
    () => ERROR_CODE_OPTIONS.map(({ labelKey, value: code }) => ({ label: t(labelKey), value: code })),
    [t],
  );

  const options = useMemo<SearchSelectOption[]>(() => {
    const trimmed = query.trim();
    const lowered = trimmed.toLowerCase();
    const matches = errorCodeOptions.filter((option) => option.label.toLowerCase().includes(lowered));
    const isKnownCode = errorCodeOptions.some(
      (option) => option.value === trimmed || option.label.toLowerCase() === lowered,
    );
    if (trimmed === "" || isKnownCode) return matches;
    return [...matches, { label: t("logs.request.filters.customErrorCode", { code: trimmed }), value: trimmed }];
  }, [query, t, errorCodeOptions]);

  const selected = useMemo<SearchSelectOption | null>(() => {
    if (value === "") return null;
    return errorCodeOptions.find((option) => option.value === value) ?? { label: value, value };
  }, [value, errorCodeOptions]);

  const items = useMemo<SearchSelectOption[]>(() => {
    if (selected === null) return options;
    if (options.some((option) => option.value === selected.value)) return options;
    return [selected, ...options];
  }, [options, selected]);

  return (
    <DataTableFilterField label={t("logs.request.filters.errorCode")}>
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
          placeholder={t("logs.request.filters.errorCodePlaceholder")}
          showClear={value !== ""}
          className="w-full"
        />
        <ComboboxContent>
          <ComboboxEmpty>{t("logs.request.filters.noErrorCodes")}</ComboboxEmpty>
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
  const { t } = useTranslation();
  const valueOf = (id: string): string => asString(get(id));
  const setter = (id: string) => (next: string | undefined) => set(id, next);
  const withLabel = ({ value, labelKey }: FilterItem) => ({ value, label: t(labelKey) });
  const spanTypeItems = SPAN_TYPE_FILTER_ITEMS.map(withLabel);
  const statusItems = STATUS_FILTER_ITEMS.map(withLabel);
  const cacheItems = CACHE_FILTER_ITEMS.map(withLabel);
  const credentialItems = CREDENTIAL_FILTER_ITEMS.map(withLabel);

  return (
    <>
      <TeamFilterField
        value={valueOf(LOG_FILTER_IDS.TEAM_ID)}
        onChange={setter(LOG_FILTER_IDS.TEAM_ID)}
        teams={teams}
      />

      <DataTableFilterField label={t("logs.request.filters.spanType")}>
        <Select
          items={spanTypeItems}
          value={valueOf(LOG_FILTER_IDS.SPAN_TYPE) === "" ? ALL_VALUE : valueOf(LOG_FILTER_IDS.SPAN_TYPE)}
          onValueChange={(next) =>
            set(LOG_FILTER_IDS.SPAN_TYPE, next === null || next === ALL_VALUE ? undefined : next)
          }
        >
          <SelectTrigger className="w-full">
            <SelectValue placeholder={t("logs.request.filters.allTypes")} />
          </SelectTrigger>
          <SelectContent>
            {spanTypeItems.map((item) => (
              <SelectItem key={item.value} value={item.value}>
                {item.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </DataTableFilterField>

      <DataTableFilterField label={t("logs.request.filters.status")}>
        <Select
          items={statusItems}
          value={valueOf(LOG_FILTER_IDS.STATUS) === "" ? ALL_VALUE : valueOf(LOG_FILTER_IDS.STATUS)}
          onValueChange={(next) => set(LOG_FILTER_IDS.STATUS, next === null || next === ALL_VALUE ? undefined : next)}
        >
          <SelectTrigger className="w-full">
            <SelectValue placeholder={t("logs.request.filters.allStatuses")} />
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

      <DataTableFilterField label={t("logs.request.filters.cache")}>
        <Select
          items={cacheItems}
          value={valueOf(LOG_FILTER_IDS.CACHE_STATUS) === "" ? ALL_VALUE : valueOf(LOG_FILTER_IDS.CACHE_STATUS)}
          onValueChange={(next) =>
            set(LOG_FILTER_IDS.CACHE_STATUS, next === null || next === ALL_VALUE ? undefined : next)
          }
        >
          <SelectTrigger className="w-full">
            <SelectValue placeholder={t("logs.request.filters.allRequests")} />
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

      <DataTableFilterField label={t("logs.request.filters.credential")}>
        <Select
          items={credentialItems}
          value={valueOf(LOG_FILTER_IDS.CREDENTIAL) === "" ? ALL_VALUE : valueOf(LOG_FILTER_IDS.CREDENTIAL)}
          onValueChange={(next) =>
            set(LOG_FILTER_IDS.CREDENTIAL, next === null || next === ALL_VALUE ? undefined : next)
          }
        >
          <SelectTrigger className="w-full">
            <SelectValue placeholder={t("logs.request.filters.allCredentials")} />
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
        excludeValue={valueOf(LOG_FILTER_IDS.EXCLUDE_KEY_ALIAS)}
        onChange={setter(LOG_FILTER_IDS.KEY_ALIAS)}
        onExcludeChange={setter(LOG_FILTER_IDS.EXCLUDE_KEY_ALIAS)}
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

      <DataTableFilterField label={t("logs.request.filters.errorMessage")}>
        <Input
          value={valueOf(LOG_FILTER_IDS.ERROR_MESSAGE)}
          onChange={(event) => set(LOG_FILTER_IDS.ERROR_MESSAGE, emptyToUndefined(event.target.value))}
          placeholder={t("logs.request.filters.errorMessagePlaceholder")}
        />
      </DataTableFilterField>

      <DataTableFilterField label={t("logs.request.filters.keyHash")}>
        <Input
          value={valueOf(LOG_FILTER_IDS.KEY_HASH)}
          onChange={(event) => set(LOG_FILTER_IDS.KEY_HASH, emptyToUndefined(event.target.value))}
          placeholder={t("logs.request.filters.keyHashPlaceholder")}
        />
      </DataTableFilterField>

      <DataTableFilterField label={t("logs.request.filters.sessionId")}>
        <Input
          value={valueOf(LOG_FILTER_IDS.SESSION_ID)}
          onChange={(event) => set(LOG_FILTER_IDS.SESSION_ID, emptyToUndefined(event.target.value))}
          placeholder={t("logs.request.filters.sessionIdPlaceholder")}
        />
      </DataTableFilterField>

      <ModelFilterField value={valueOf(LOG_FILTER_IDS.MODEL_ID)} onChange={setter(LOG_FILTER_IDS.MODEL_ID)} />

      <DataTableFilterField label={t("logs.request.filters.publicModelOrSearchTool")}>
        <Input
          value={valueOf(LOG_FILTER_IDS.PUBLIC_MODEL_OR_SEARCH_TOOL)}
          onChange={(event) => set(LOG_FILTER_IDS.PUBLIC_MODEL_OR_SEARCH_TOOL, emptyToUndefined(event.target.value))}
          placeholder={t("logs.request.filters.publicModelOrSearchToolPlaceholder")}
        />
      </DataTableFilterField>
    </>
  );
}
