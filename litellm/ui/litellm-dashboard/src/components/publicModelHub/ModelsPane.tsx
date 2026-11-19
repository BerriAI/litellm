"use client";

import type { ColumnDef } from "@tanstack/react-table";
import { ArrowDownWideNarrow, ArrowUpNarrowWide, Boxes } from "lucide-react";
import type { MouseEvent } from "react";
import { useTranslation } from "react-i18next";

import type { ModelGroupInfo } from "@/components/PublicModelHubTableColumns";
import { DataTable, DataTableSortHeader } from "@/components/shared/DataTable";
import { ButtonGroup } from "@/components/ui/button-group";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
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
const filterSection = "min-w-0 border-b border-b-border pb-[22px] last:border-b-0";
const filterLegend = "mb-3 text-xs leading-[1.2] font-semibold";
const modeButton =
  "h-auto gap-[normal] rounded-none border border-transparent bg-transparent px-3 py-[7px] text-xs leading-[1.6] font-normal text-muted-foreground hover:border-border hover:bg-background hover:text-foreground focus-visible:border-transparent focus-visible:ring-0 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-foreground active:not-aria-[haspopup]:translate-y-0 aria-pressed:border-(--catalog-rule-strong) aria-pressed:bg-muted aria-pressed:text-foreground aria-pressed:hover:border-border aria-pressed:focus-visible:border-(--catalog-rule-strong) focus-visible:hover:border-border dark:hover:bg-background dark:aria-pressed:hover:bg-muted max-[640px]:px-[9px]";
const tableCell = "h-[66px] px-4 py-3 text-[13px] leading-[1.6] font-normal text-foreground max-[640px]:px-3";
const tableHeading = "h-auto py-3.5 text-[11px] font-medium text-muted-foreground";

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
        <SelectTrigger
          size="sm"
          className="w-auto gap-1.5 text-xs leading-[1.6] font-normal shadow-none"
          aria-label={t("publicModelHub.sortControl.sortBy")}
        >
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
              className="text-xs leading-[1.6] font-normal shadow-none dark:bg-input/30 dark:hover:bg-input/50"
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
    <fieldset data-slot="catalog-filter-section" className={filterSection}>
      <legend className={filterLegend}>{label}</legend>
      <div className="flex flex-col gap-[11px]">
        {options.map((option) => (
          <label
            key={option.value}
            className="flex cursor-pointer items-center gap-2.5 text-xs leading-[1.6] font-normal"
          >
            <Input
              type="checkbox"
              className="size-3.5 shrink-0 appearance-auto rounded-none border-0 bg-transparent p-0 accent-foreground shadow-none focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-foreground focus-visible:ring-0 dark:bg-transparent"
              checked={values.includes(option.value)}
              onChange={(event) =>
                onChange(
                  event.target.checked ? [...values, option.value] : values.filter((value) => value !== option.value),
                )
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
      <fieldset data-slot="catalog-filter-section" className={filterSection}>
        <legend className={filterLegend}>{t("publicModelHub.catalog.provider")}</legend>
        <div className="relative">
          <svg
            className="pointer-events-none absolute top-[19px] left-[9px] z-raised -translate-y-1/2 text-muted-foreground"
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
            className="min-h-[38px] gap-[5px] rounded-[7px] border bg-card py-2 pr-[25px] pl-[29px] text-xs leading-[1.6] font-normal text-foreground shadow-none dark:bg-card [&_input]:leading-[1.35] [&_input::placeholder]:text-foreground [&_input::placeholder]:opacity-100"
            contentClassName={catalogTypography}
          />
          <svg
            className="pointer-events-none absolute top-[19px] right-[9px] z-raised -translate-y-1/2 text-muted-foreground"
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
    <div
      data-slot="catalog-modes"
      className="flex flex-wrap items-center gap-[7px] border-b border-b-border pt-1.5 pb-[18px] text-base leading-[1.6] max-[640px]:gap-1"
    >
      <span className="mr-[9px] text-[11px] text-muted-foreground">{t("publicModelHub.catalog.mode")}</span>
      <Button
        className={modeButton}
        variant="ghost"
        aria-pressed={models.modeValues.length === 0}
        onClick={() => models.onModesChange([])}
      >
        {t("publicModelHub.catalog.allModes")}
      </Button>
      <ToggleGroup
        className="flex-wrap gap-[7px] max-[640px]:gap-1"
        multiple
        value={models.modeValues}
        onValueChange={models.onModesChange}
        aria-label={t("publicModelHub.catalog.mode")}
      >
        {modes.map((mode) => (
          <ToggleGroupItem className={modeButton} key={mode} value={mode}>
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
      data-slot="catalog-model-link"
      className="flex min-h-0 min-w-0 items-center gap-3 rounded-sm text-sm leading-[1.6] outline-none hover:text-primary focus-visible:ring-2 focus-visible:ring-ring"
      onClick={follow}
    >
      <svg className="size-3.5 shrink-0 text-muted-foreground" viewBox="0 0 16 16" fill="none" aria-hidden="true">
        <path d="m6 4 4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      <ProviderAvatar
        provider={model.providers?.[0] ?? ""}
        className="size-[30px] shrink-0 gap-[3px] rounded-sm ring-0"
      />
      <span className="flex min-w-0 flex-col gap-[3px]">
        <span className="leading-[1.4] font-medium whitespace-normal wrap-break-word">{model.model_group}</span>
        <span className="text-[11px] leading-[1.3] font-normal whitespace-normal text-muted-foreground">
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
      meta: {
        skeleton: "twoLine",
        className: `${tableCell} text-sm/[1.6] font-medium`,
        headerClassName: `${tableHeading} w-[38%]`,
      },
      size: 245,
      sortDescFirst: false,
      header: ({ column }) => (
        <DataTableSortHeader
          className="[font:inherit] text-inherit"
          column={column}
          title={t("publicModelHub.catalog.model", { defaultValue: "Model" })}
        />
      ),
      cell: ({ row }) => <ModelLink model={row.original} selected={selected} onSelect={onSelect} />,
    },
    {
      accessorKey: "max_input_tokens",
      meta: { className: `${tableCell} hidden sm:table-cell tabular-nums`, headerClassName: tableHeading },
      size: 140,
      sortDescFirst: true,
      header: ({ column }) => (
        <DataTableSortHeader
          column={column}
          className="[font:inherit] text-inherit"
          title={t("publicModelHub.catalog.inputLimit", { defaultValue: "Input limit" })}
        />
      ),
      cell: ({ row }) => (row.original.max_input_tokens === 0 ? "0" : formatTokens(row.original.max_input_tokens)),
    },
    {
      accessorKey: "input_cost_per_token",
      meta: { className: `${tableCell} text-(--catalog-secondary) tabular-nums`, headerClassName: tableHeading },
      size: 130,
      sortDescFirst: false,
      header: ({ column }) => (
        <DataTableSortHeader
          column={column}
          className="[font:inherit] text-inherit"
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
      meta: { className: `${tableCell} text-(--catalog-secondary) tabular-nums`, headerClassName: tableHeading },
      size: 130,
      sortDescFirst: false,
      header: ({ column }) => (
        <DataTableSortHeader
          column={column}
          className="[font:inherit] text-inherit"
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
    <div
      data-slot="catalog-layout"
      className="grid min-w-0 grid-cols-[218px_minmax(0,1fr)] items-start gap-[34px] pt-6 catalog-narrow:grid-cols-1 catalog-narrow:gap-5 catalog-narrow:pt-[22px]"
    >
      <ModelFilters hub={hub} />
      <div data-slot="catalog-results" className="flex min-w-0 flex-col gap-0">
        <div data-slot="catalog-toolbar" className="mb-4 flex flex-wrap items-center gap-3">
          <SearchInput
            value={models.searchValue}
            onChange={models.onSearchChange}
            placeholder={t("publicModelHub.catalog.searchModels", { defaultValue: "Search models" })}
            showShortcut
          />
        </div>
        <ModeFilters hub={hub} />
        <div className="flex flex-wrap items-center justify-between gap-3 py-4">
          <div
            data-slot="catalog-count"
            className="text-xs leading-[1.6] font-normal text-muted-foreground tabular-nums"
            aria-live="polite"
          >
            {models.isFetching || models.isLoading
              ? t("publicModelHub.catalog.loadingModels", { defaultValue: "Loading models…" })
              : !models.error && t("publicModelHub.catalog.modelCount", countOptions)}
          </div>
          <SortControl models={models} />
        </div>
        {models.error && models.rows.length > 0 && (
          <LoadFailed noun={t("publicModelHub.modelsPane.noun")} onRetry={models.refetch} />
        )}
        <div
          data-slot="catalog-table"
          className="overflow-hidden rounded-lg bg-card [&_table]:min-w-[640px] [&_table]:text-base [&_table]:leading-[1.6] [&_thead]:bg-transparent [&_thead_tr]:bg-transparent [&_thead_tr]:border-border [&_[data-slot=table-pagination]]:flex-wrap [&_[data-slot=table-pagination]]:gap-3 [&_[data-slot=table-pagination]]:px-4 [&_[data-slot=table-pagination]]:py-3 [&_[data-slot=table-pagination]]:text-xs/[1.4285714286] [&_[data-slot=table-pagination-summary]]:flex-wrap [&_[data-slot=table-pagination-summary]]:gap-3 [&_[data-slot=table-pagination]_:is(p,span,button)]:text-xs/[1.6] [&_[data-slot=table-pagination]_:is(p,span,button)]:font-normal [&_[data-slot=table-pagination]_:is(p,span,button)]:text-muted-foreground"
          aria-busy={models.isFetching || models.isLoading}
        >
          <DataTable
            rowClassName={() => "border-border hover:bg-(--catalog-hover) hover:shadow-[inset_3px_0_0_var(--primary)]"}
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
