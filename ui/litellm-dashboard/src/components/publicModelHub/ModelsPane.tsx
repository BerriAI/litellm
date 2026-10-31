"use client";

import { ArrowDownWideNarrow, ArrowUpNarrowWide, Boxes, ChevronLeft, ChevronRight } from "lucide-react";

import type { ModelGroupInfo } from "@/components/PublicModelHubTableColumns";
import { Button } from "@/components/ui/button";
import { ButtonGroup } from "@/components/ui/button-group";
import { ItemActions, ItemContent, ItemDescription, ItemMedia } from "@/components/ui/item";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Toggle } from "@/components/ui/toggle";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

import { FilterChip } from "./FilterChip";
import { featureFacetLabel, firstPlus, formatPerMillion, formatTokens, modeLabel, providerName } from "./hubFormat";
import { HubMasterDetail } from "./HubMasterDetail";
import { NoResults } from "./HubEmpty";
import { Joined, ProviderAvatar, RowTitle, SearchInput } from "./HubParts";
import type { HubData } from "./useHubData";
import { ModelInspector } from "./ModelInspector";

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

function ModelsPager({ models }: { models: Models }) {
  const { pagination, rowCount, rows, onPaginationChange } = models;
  if (!rowCount) return null;
  const start = pagination.pageIndex * pagination.pageSize + 1;
  const end = start + rows.length - 1;
  const go = (pageIndex: number) => onPaginationChange({ ...pagination, pageIndex });
  return (
    <div className="flex items-center justify-between gap-2 px-6 pb-3 text-xs text-muted-foreground tabular-nums">
      <span>
        {start}–{end} of {rowCount}
      </span>
      <ButtonGroup>
        <Button
          variant="outline"
          size="icon-sm"
          aria-label="Previous page"
          disabled={pagination.pageIndex === 0}
          onClick={() => go(pagination.pageIndex - 1)}
        >
          <ChevronLeft />
        </Button>
        <Button
          variant="outline"
          size="icon-sm"
          aria-label="Next page"
          disabled={start + pagination.pageSize > rowCount}
          onClick={() => go(pagination.pageIndex + 1)}
        >
          <ChevronRight />
        </Button>
      </ButtonGroup>
    </div>
  );
}

function SortControl({ models }: { models: Models }) {
  const field = models.sorting[0]?.id ?? "model_group";
  const desc = models.sorting[0]?.desc ?? false;
  const pick = (value: string | null) => {
    const sort = SORTS.find((option) => option.value === value) ?? SORTS[0];
    models.onSortingChange([{ id: sort.value, desc: sort.desc }]);
  };
  return (
    <ButtonGroup>
      <Select items={SORTS} value={field} onValueChange={pick}>
        <SelectTrigger size="sm" className="w-auto gap-1.5" aria-label="Sort by">
          <span className="hidden text-muted-foreground sm:inline">Sort</span>
          <SelectValue />
        </SelectTrigger>
        <SelectContent align="end" className="w-auto min-w-(--anchor-width)">
          {SORTS.map((sort) => (
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
              aria-label="Sort descending"
              pressed={desc}
              onPressedChange={(pressed) => models.onSortingChange([{ id: field, desc: pressed }])}
              className="dark:bg-input/30 dark:hover:bg-input/50"
            />
          }
        >
          {desc ? <ArrowDownWideNarrow /> : <ArrowUpNarrowWide />}
        </TooltipTrigger>
        <TooltipContent>Sort descending</TooltipContent>
      </Tooltip>
    </ButtonGroup>
  );
}

function ModelsToolbar({ hub }: { hub: HubData }) {
  const { models, facets } = hub;
  return (
    <>
      <div className="flex items-center gap-2">
        <SearchInput value={models.searchValue} onChange={models.onSearchChange} placeholder="Search models" />
        <SortControl models={models} />
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <FilterChip
          label="Provider"
          noun="providers"
          options={facets.providers.map((p) => ({ value: p, label: providerName(p), provider: p }))}
          value={models.providerValues}
          onChange={models.onProvidersChange}
        />
        <FilterChip
          label="Mode"
          noun="modes"
          options={facets.modes.map((m) => ({ value: m, label: modeLabel(m) }))}
          value={models.modeValues}
          onChange={models.onModesChange}
        />
        <FilterChip
          label="Features"
          noun="features"
          options={facets.features.map((f) => ({ value: f, label: featureFacetLabel(f) }))}
          value={models.featureValues}
          onChange={models.onFeaturesChange}
        />
      </div>
    </>
  );
}

const prices = (model: ModelGroupInfo) => [
  model.input_cost_per_token ? `${formatPerMillion(model.input_cost_per_token)} / 1M in` : null,
  model.output_cost_per_token ? `${formatPerMillion(model.output_cost_per_token)} / 1M out` : null,
];

function ModelRow({ model }: { model: ModelGroupInfo }) {
  const [input, output] = prices(model);
  const priced = Boolean(input || output);
  return (
    <>
      <ItemMedia variant="icon" className="size-5">
        <ProviderAvatar provider={model.providers?.[0] ?? ""} className="size-5" />
      </ItemMedia>
      <ItemContent className="min-w-0 gap-0.5">
        <RowTitle>{model.model_group}</RowTitle>
        <ItemDescription>
          <Joined
            parts={[
              firstPlus((model.providers ?? []).map(providerName)),
              model.mode ? modeLabel(model.mode) : null,
              model.max_input_tokens ? `${formatTokens(model.max_input_tokens)} context` : null,
            ]}
          />
        </ItemDescription>
        {priced && (
          <ItemDescription className="tabular-nums sm:hidden">
            <Joined parts={[input, output]} />
          </ItemDescription>
        )}
      </ItemContent>
      {priced && (
        <ItemActions className="hidden flex-col items-end gap-0.5 text-xs whitespace-nowrap tabular-nums sm:flex">
          {input && <span>{input}</span>}
          {output && <span className="text-muted-foreground">{output}</span>}
        </ItemActions>
      )}
    </>
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
  const { models } = hub;
  const clear = () => {
    models.onSearchChange("");
    models.onColumnFiltersChange([]);
  };
  return (
    <HubMasterDetail
      rows={models.rows}
      loading={models.isLoading}
      selectedKey={selected}
      getKey={(model) => model.model_group}
      onSelect={onSelect}
      empty={
        <NoResults
          icon={<Boxes />}
          noun="models"
          query={models.searchValue}
          onClear={models.hasActiveQuery ? clear : undefined}
        />
      }
      toolbar={<ModelsToolbar hub={hub} />}
      renderItem={(model) => <ModelRow model={model} />}
      renderDetail={(model) => <ModelInspector model={model} />}
      footer={<ModelsPager models={models} />}
    />
  );
}
