"use client";

import { ArrowDownWideNarrow, ArrowUpNarrowWide, Boxes, ChevronLeft, ChevronRight } from "lucide-react";
import { useTranslation } from "react-i18next";

import type { ModelGroupInfo } from "@/components/PublicModelHubTableColumns";
import { Button } from "@/components/ui/button";
import { ButtonGroup } from "@/components/ui/button-group";
import { ItemActions, ItemContent, ItemDescription, ItemMedia } from "@/components/ui/item";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Toggle } from "@/components/ui/toggle";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { displayMode } from "@/lib/decisionModels";
import i18n from "@/i18n/i18n";

import { FilterChip } from "./FilterChip";
import { featureFacetLabel, firstPlus, formatPerMillion, formatTokens, modeLabel, providerName } from "./hubFormat";
import { HubMasterDetail } from "./HubMasterDetail";
import { LoadFailed, NoResults } from "./HubEmpty";
import { Joined, ProviderAvatar, RowTitle, SearchInput } from "./HubParts";
import { type HubData, useOpenedModel } from "./useHubData";
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
  const { t } = useTranslation();
  const { pagination, rowCount, rows, onPaginationChange } = models;
  if (!rowCount) return null;
  const start = pagination.pageIndex * pagination.pageSize + 1;
  const end = start + rows.length - 1;
  const go = (pageIndex: number) => onPaginationChange({ ...pagination, pageIndex });
  return (
    <div className="flex items-center justify-between gap-2 px-6 pb-3 text-xs text-muted-foreground tabular-nums">
      <span>{t("publicModelHub.modelsPager.range", { start, end, total: rowCount })}</span>
      <ButtonGroup>
        <Button
          variant="outline"
          size="icon-sm"
          aria-label={t("publicModelHub.modelsPager.previous")}
          disabled={pagination.pageIndex === 0}
          onClick={() => go(pagination.pageIndex - 1)}
        >
          <ChevronLeft />
        </Button>
        <Button
          variant="outline"
          size="icon-sm"
          aria-label={t("publicModelHub.modelsPager.next")}
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
  const { t } = useTranslation();
  const field = models.sorting[0]?.id ?? "model_group";
  const desc = models.sorting[0]?.desc ?? false;
  const pick = (value: string | null) => {
    const sort = SORTS.find((option) => option.value === value) ?? SORTS[0];
    models.onSortingChange([{ id: sort.value, desc: sort.desc }]);
  };
  return (
    <ButtonGroup>
      <Select items={SORTS} value={field} onValueChange={pick}>
        <SelectTrigger size="sm" className="w-auto gap-1.5" aria-label={t("publicModelHub.sortControl.sortBy")}>
          <span className="hidden text-muted-foreground sm:inline">{t("publicModelHub.sortControl.sort")}</span>
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

function ModelsToolbar({ hub }: { hub: HubData }) {
  const { t } = useTranslation();
  const { models, facets } = hub;
  return (
    <>
      <div className="flex items-center gap-2">
        <SearchInput
          value={models.searchValue}
          onChange={models.onSearchChange}
          placeholder={t("publicModelHub.modelsToolbar.search")}
        />
        <SortControl models={models} />
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <FilterChip
          label={t("publicModelHub.modelsToolbar.provider")}
          noun={t("publicModelHub.modelsToolbar.providerNoun")}
          options={facets.providers.map((p) => ({ value: p, label: providerName(p), provider: p }))}
          value={models.providerValues}
          onChange={models.onProvidersChange}
        />
        <FilterChip
          label={t("publicModelHub.modelsToolbar.mode")}
          noun={t("publicModelHub.modelsToolbar.modeNoun")}
          options={[...new Set(facets.modes.map((m) => displayMode(m) ?? m))].map((m) => ({
            value: m,
            label: modeLabel(m),
          }))}
          value={models.modeValues}
          onChange={models.onModesChange}
        />
        <FilterChip
          label={t("publicModelHub.modelsToolbar.features")}
          noun={t("publicModelHub.modelsToolbar.featureNoun")}
          options={facets.features.map((f) => ({ value: f, label: featureFacetLabel(f) }))}
          value={models.featureValues}
          onChange={models.onFeaturesChange}
        />
      </div>
    </>
  );
}

const price = (cost: number | null | undefined, direction: "input" | "output") => {
  if (cost === 0) return i18n.t("publicModelHub.modelPrices.freeDirection", { direction });
  return cost
    ? i18n.t("publicModelHub.modelPrices.perMillion", {
        price: formatPerMillion(cost),
        direction:
          direction === "input" ? i18n.t("publicModelHub.modelPrices.in") : i18n.t("publicModelHub.modelPrices.out"),
      })
    : null;
};

const prices = (model: ModelGroupInfo) =>
  model.input_cost_per_token === 0 && model.output_cost_per_token === 0
    ? [i18n.t("publicModelHub.modelPrices.free"), null]
    : [price(model.input_cost_per_token, "input"), price(model.output_cost_per_token, "output")];

function ModelRow({ model }: { model: ModelGroupInfo }) {
  const { t } = useTranslation();
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
              model.max_input_tokens
                ? t("publicModelHub.modelRow.context", { tokens: formatTokens(model.max_input_tokens) })
                : null,
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
  const { t } = useTranslation();
  const { models } = hub;
  const opened = useOpenedModel(selected, models.rows, !models.isLoading);
  const clear = () => {
    models.onSearchChange("");
    models.onColumnFiltersChange([]);
  };
  return (
    <HubMasterDetail
      rows={models.rows}
      loading={models.isLoading}
      selectedKey={selected}
      opened={opened.model}
      openPending={opened.pending}
      getKey={(model) => model.model_group}
      onSelect={onSelect}
      empty={
        models.error ? (
          <LoadFailed noun={t("publicModelHub.modelsPane.noun")} onRetry={models.refetch} />
        ) : (
          <NoResults
            icon={<Boxes />}
            noun={t("publicModelHub.modelsPane.noun")}
            query={models.searchValue}
            onClear={models.hasActiveQuery ? clear : undefined}
          />
        )
      }
      toolbar={<ModelsToolbar hub={hub} />}
      renderItem={(model) => <ModelRow model={model} />}
      renderDetail={(model) => <ModelInspector model={model} />}
      footer={<ModelsPager models={models} />}
    />
  );
}
