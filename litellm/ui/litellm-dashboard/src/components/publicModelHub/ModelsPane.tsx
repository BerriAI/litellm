"use client";

import type { ColumnDef } from "@tanstack/react-table";
import { ArrowDownWideNarrow, ArrowUpNarrowWide, Boxes, ChevronRight } from "lucide-react";
import type { MouseEvent } from "react";
import { useTranslation } from "react-i18next";

import type { ModelGroupInfo } from "@/components/PublicModelHubTableColumns";
import { DataTable, DataTableSortHeader } from "@/components/shared/DataTable";
import { ButtonGroup } from "@/components/ui/button-group";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Toggle } from "@/components/ui/toggle";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { displayMode } from "@/lib/decisionModels";

import { CatalogDetails } from "./CatalogDetails";
import { CatalogFilters } from "./CatalogFilters";
import { featureFacetLabel, formatPerMillion, formatTokens, modeLabel, providerName } from "./hubFormat";
import { LoadFailed, NoResults, NotListed } from "./HubEmpty";
import { ProviderAvatar, SearchInput } from "./HubParts";
import { type HubData, useOpenedModel } from "./useHubData";
import { useOpenHref } from "./useHubUrlState";
import { ModelInspector } from "./ModelInspector";
import styles from "./catalog.module.css";
import { catalogTypography } from "./fonts";

const SORTS = [
  { value: "model_group", label: "Name", desc: false },
  { value: "providers", label: "Provider", desc: false },
  { value: "mode", label: "Mode", desc: false },
  { value: "input_cost_per_token", label: "Input price", desc: false },
  { value: "output_cost_per_token", label: "Output price", desc: false },
  { value: "max_input_tokens", label: "Context", desc: true },
  { value: "max_output_tokens", label: "Max output", desc: true },
  { value: "rpm", label: "Rate limit", desc: true },
];

type Models = HubData["models"];

function SortControl({ models }: { models: Models }) {
  const { t } = useTranslation();
  const options = SORTS.map((sort) => ({
    ...sort,
    label: t(`publicModelHub.catalog.sort.${sort.value}`, { defaultValue: sort.label }),
  }));
  const field = models.sorting[0]?.id ?? "model_group";
  const desc = models.sorting[0]?.desc ?? false;
  const pick = (value: string | null) => {
    const sort = SORTS.find((option) => option.value === value) ?? SORTS[0];
    models.onSortingChange([{ id: sort.value, desc: sort.desc }]);
  };
  return (
    <ButtonGroup>
      <Select items={options} value={field} onValueChange={pick}>
        <SelectTrigger size="sm" className="w-auto gap-1.5" aria-label={t("publicModelHub.sortControl.sortBy")}>
          <span className="hidden text-muted-foreground sm:inline">{t("publicModelHub.sortControl.sort")}</span>
          <SelectValue />
        </SelectTrigger>
        <SelectContent align="end" className={`${catalogTypography} w-auto min-w-(--anchor-width)`}>
          {options.map((sort) => (
            <SelectItem key={sort.value} value={sort.value}>
              {sort.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <Tooltip>
        <TooltipTrigger
          render={
            <Toggle
              variant="outline"
              size="sm"
              aria-label={t("publicModelHub.sortControl.descending")}
              pressed={desc}
              onPressedChange={(pressed) => models.onSortingChange([{ id: field, desc: pressed }])}
              className="dark:bg-input/30 dark:hover:bg-input/50"
            />
          }
        >
          {desc ? <ArrowDownWideNarrow /> : <ArrowUpNarrowWide />}
        </TooltipTrigger>
        <TooltipContent>{t("publicModelHub.sortControl.descending")}</TooltipContent>
      </Tooltip>
    </ButtonGroup>
  );
}

interface FilterOption {
  value: string;
  label: string;
}

function CatalogFilter({
  label,
  options,
  values,
  onChange,
}: {
  label: string;
  options: FilterOption[];
  values: string[];
  onChange: (values: string[]) => void;
}) {
  return (
    <fieldset data-slot="catalog-filter-section" className="min-w-0">
      <legend className="mb-3 text-sm font-medium">{label}</legend>
      <div className="flex flex-col gap-3">
        {options.map((option) => (
          <label key={option.value} className="flex cursor-pointer items-center gap-2.5 text-sm">
            <Checkbox
              checked={values.includes(option.value)}
              onCheckedChange={(checked) =>
                onChange(checked ? [...values, option.value] : values.filter((value) => value !== option.value))
              }
            />
            <span className="min-w-0 break-words">{option.label}</span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}

function ModelFilters({ hub }: { hub: HubData }) {
  const { t } = useTranslation();
  const { models, facets } = hub;
  return (
    <CatalogFilters count={models.columnFilters.length} onClear={() => models.onColumnFiltersChange([])}>
      <CatalogFilter
        label={t("publicModelHub.catalog.capabilities", { defaultValue: "Capabilities" })}
        options={facets.features.map((feature) => ({ value: feature, label: featureFacetLabel(feature) }))}
        values={models.featureValues}
        onChange={models.onFeaturesChange}
      />
      <fieldset data-slot="catalog-filter-section" className="min-w-0">
        <legend className="mb-3 text-sm font-medium">{t("publicModelHub.catalog.provider")}</legend>
        <div className={styles.providerControl}>
          <svg
            className={styles.providerControlIcon}
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
            aria-hidden="true"
          >
            <rect x="3" y="3" width="7" height="7" rx="1.5" />
            <rect x="14" y="3" width="7" height="7" rx="1.5" />
            <rect x="3" y="14" width="7" height="7" rx="1.5" />
            <rect x="14" y="14" width="7" height="7" rx="1.5" />
          </svg>
          <MultiSelect
            options={facets.providers.map((provider) => ({
              value: provider,
              label: providerName(provider),
              icon: <ProviderAvatar provider={provider} />,
            }))}
            value={models.providerValues}
            onValueChange={models.onProvidersChange}
            placeholder={t("publicModelHub.catalog.allProviders")}
            className={styles.providerSelect}
            contentClassName={catalogTypography}
          />
          <svg
            className={styles.providerControlChevron}
            width="12"
            height="12"
            viewBox="0 0 20 20"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            aria-hidden="true"
          >
            <path d="m6 8 4 4 4-4" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
      </fieldset>
    </CatalogFilters>
  );
}

function ModeFilters({ hub }: { hub: HubData }) {
  const { t } = useTranslation();
  const { models, facets } = hub;
  const modes = [...new Set(facets.modes.map((mode) => displayMode(mode) ?? mode))];
  return (
    <div className={styles.modeFilters}>
      <span>{t("publicModelHub.catalog.mode")}</span>
      <Button variant="ghost" aria-pressed={models.modeValues.length === 0} onClick={() => models.onModesChange([])}>
        {t("publicModelHub.catalog.allModes")}
      </Button>
      <ToggleGroup
        multiple
        value={models.modeValues}
        onValueChange={models.onModesChange}
        aria-label={t("publicModelHub.catalog.mode")}
      >
        {modes.map((mode) => (
          <ToggleGroupItem key={mode} value={mode}>
            {modeLabel(mode)}
          </ToggleGroupItem>
        ))}
      </ToggleGroup>
    </div>
  );
}

function ModelLink({
  model,
  selected,
  onSelect,
}: {
  model: ModelGroupInfo;
  selected: string | null;
  onSelect: (key: string | null) => void;
}) {
  const hrefFor = useOpenHref();
  const follow = (event: MouseEvent<HTMLAnchorElement>) => {
    if (event.button !== 0 || event.defaultPrevented) return;
    if (event.metaKey || event.ctrlKey) return;
    if (event.shiftKey || event.altKey) return;
    event.preventDefault();
    onSelect(model.model_group);
  };
  return (
    <a
      href={hrefFor(model.model_group)}
      aria-current={selected === model.model_group ? "true" : undefined}
      className={`${styles.modelLink} flex min-w-0 items-center gap-3 rounded-sm outline-none hover:text-primary focus-visible:ring-2 focus-visible:ring-ring`}
      onClick={follow}
    >
      <ChevronRight className="size-3.5 shrink-0 text-muted-foreground" aria-hidden="true" />
      <ProviderAvatar provider={model.providers?.[0] ?? ""} className="size-[30px] shrink-0 rounded-sm ring-0" />
      <span className="flex min-w-0 flex-col gap-1">
        <span className="font-medium whitespace-normal wrap-anywhere">{model.model_group}</span>
        <span className="text-xs whitespace-normal text-muted-foreground">
          {(model.providers ?? []).map(providerName).join(", ")}
        </span>
      </span>
    </a>
  );
}

export function ModelsPane({
  hub,
  selected,
  onSelect,
}: {
  hub: HubData;
  selected: string | null;
  onSelect: (key: string | null) => void;
}) {
  const { t } = useTranslation();
  const { models } = hub;
  const opened = useOpenedModel(selected, models.rows, !models.isLoading);
  const detailsLoading = opened.model === undefined && (models.isLoading || opened.pending);
  const missingName = opened.model === undefined && !detailsLoading ? selected : null;
  const countOptions = {
    count: models.rowCount,
    defaultValue_one: "{{count}} model",
    defaultValue_other: "{{count}} models",
    defaultValue: "{{count}} models",
  };
  const columns: ColumnDef<ModelGroupInfo>[] = [
    {
      accessorKey: "model_group",
      meta: { skeleton: "twoLine" },
      size: 245,
      sortDescFirst: false,
      header: ({ column }) => (
        <DataTableSortHeader column={column} title={t("publicModelHub.catalog.model", { defaultValue: "Model" })} />
      ),
      cell: ({ row }) => <ModelLink model={row.original} selected={selected} onSelect={onSelect} />,
    },
    {
      accessorKey: "max_input_tokens",
      meta: { className: "hidden sm:table-cell tabular-nums" },
      size: 140,
      sortDescFirst: true,
      header: ({ column }) => (
        <DataTableSortHeader
          column={column}
          title={t("publicModelHub.catalog.inputLimit", { defaultValue: "Input limit" })}
        />
      ),
      cell: ({ row }) => (row.original.max_input_tokens === 0 ? "0" : formatTokens(row.original.max_input_tokens)),
    },
    {
      accessorKey: "input_cost_per_token",
      meta: { className: `${styles.priceCell} tabular-nums` },
      size: 130,
      sortDescFirst: false,
      header: ({ column }) => (
        <DataTableSortHeader
          column={column}
          title={t("publicModelHub.catalog.inputPrice", { defaultValue: "Input /1M" })}
        />
      ),
      cell: ({ row }) => (
        <span>
          {formatPerMillion(row.original.input_cost_per_token)}
          {!!row.original.input_cost_per_token && t("publicModelHub.catalog.perMillion")}
        </span>
      ),
    },
    {
      accessorKey: "output_cost_per_token",
      meta: { className: `${styles.priceCell} tabular-nums` },
      size: 130,
      sortDescFirst: false,
      header: ({ column }) => (
        <DataTableSortHeader
          column={column}
          title={t("publicModelHub.catalog.outputPrice", { defaultValue: "Output /1M" })}
        />
      ),
      cell: ({ row }) => (
        <span>
          {formatPerMillion(row.original.output_cost_per_token)}
          {!!row.original.output_cost_per_token && t("publicModelHub.catalog.perMillion")}
        </span>
      ),
    },
  ];
  const clear = () => {
    models.onSearchChange("");
    models.onColumnFiltersChange([]);
  };
  return (
    <div data-slot="catalog-layout" className="grid min-w-0 gap-8 lg:grid-cols-[220px_minmax(0,1fr)]">
      <ModelFilters hub={hub} />
      <div data-slot="catalog-results" className="flex min-w-0 flex-col gap-4">
        <div data-slot="catalog-toolbar" className="flex flex-wrap items-center gap-3">
          <SearchInput
            value={models.searchValue}
            onChange={models.onSearchChange}
            placeholder={t("publicModelHub.catalog.searchModels", { defaultValue: "Search models" })}
            showShortcut
          />
        </div>
        <ModeFilters hub={hub} />
        <div className={`${styles.resultsMeta} flex flex-wrap items-center justify-between gap-3 py-4`}>
          <div data-slot="catalog-count" aria-live="polite">
            {models.isFetching || models.isLoading
              ? t("publicModelHub.catalog.loadingModels", { defaultValue: "Loading models…" })
              : !models.error && t("publicModelHub.catalog.modelCount", countOptions)}
          </div>
          <SortControl models={models} />
        </div>
        {models.error && models.rows.length > 0 && (
          <LoadFailed noun={t("publicModelHub.modelsPane.noun")} onRetry={models.refetch} />
        )}
        <div data-slot="catalog-table" aria-busy={models.isFetching || models.isLoading}>
          <DataTable
            data={models.rows}
            columns={columns}
            getRowId={(model) => model.model_group}
            sortingMode="server"
            sorting={models.sorting}
            onSortingChange={models.onSortingChange}
            paginationMode="server"
            pagination={models.pagination}
            onPaginationChange={models.onPaginationChange}
            rowCount={models.rowCount}
            filterMode="server"
            columnFilters={models.columnFilters}
            onColumnFiltersChange={models.onColumnFiltersChange}
            isLoading={models.isLoading}
            isError={Boolean(models.error)}
            errorMessage={t("publicModelHub.loadFailed.title", { noun: t("publicModelHub.modelsPane.noun") })}
            onRetry={models.refetch}
            noDataMessage={
              <NoResults
                icon={<Boxes />}
                noun={t("publicModelHub.modelsPane.noun")}
                query={models.searchValue}
                onClear={models.hasActiveQuery ? clear : undefined}
              />
            }
          />
        </div>
      </div>
      <CatalogDetails name={selected} loading={detailsLoading} onClose={() => onSelect(null)}>
        {opened.model && <ModelInspector model={opened.model} />}
        {missingName !== null && <NotListed name={missingName} />}
      </CatalogDetails>
    </div>
  );
}
