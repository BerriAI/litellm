import React, { useMemo, useState } from "react";
import { BookOpen, Plus, Sparkles } from "lucide-react";

import { useModelCostMap } from "@/app/(dashboard)/hooks/models/useModelCostMap";
import { useProviderFields } from "@/app/(dashboard)/hooks/providers/useProviderFields";
import { Logo } from "@/components/molecules/logo/Logo";
import { getProviderLogoAndName } from "@/components/provider_info_helpers";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  DECISIONS_DOCS_URL,
  DECISIONS_SELF_HOSTED_DOCS_URL,
  DecisionGardenModel,
  buildDecisionGarden,
} from "@/lib/decisionModels";
import { uiHref } from "@/utils/uiHref";

const PREVIEW_COUNT = 8;

export const addModelHref = (row: DecisionGardenModel): string => {
  const params = new URLSearchParams({ tab: "add", provider: row.provider, litellm_model: row.model });
  return `${uiHref("models-and-endpoints")}?${params.toString()}`;
};

interface DecisionModelCardProps {
  readonly row: DecisionGardenModel;
  readonly canAddModels: boolean;
}

function DecisionModelCard({ row, canAddModels }: DecisionModelCardProps) {
  const { displayName } = getProviderLogoAndName(row.provider);
  return (
    <div
      className="flex flex-col gap-3 rounded-lg border bg-background p-4"
      data-testid={`decision-model-card-${row.model}`}
    >
      <div className="flex items-center gap-2">
        <Logo provider={row.provider} className="size-5" />
        <span className="truncate text-sm font-medium text-foreground">{displayName}</span>
      </div>
      <code className="truncate text-xs text-muted-foreground" title={row.model}>
        {row.model}
      </code>
      <div className="mt-auto flex items-center justify-between gap-2">
        <a
          href={row.route === "decisions" ? DECISIONS_DOCS_URL : DECISIONS_SELF_HOSTED_DOCS_URL}
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs text-muted-foreground underline"
        >
          {row.route === "decisions" ? "/v1/decisions docs" : "Self-hosted, own route"}
        </a>
        {canAddModels && (
          <Button size="xs" variant="outline" nativeButton={false} role="link" render={<a href={addModelHref(row)} />}>
            <Plus />
            Add model
          </Button>
        )}
      </div>
    </div>
  );
}

interface DecisionModelGardenProps {
  readonly canAddModels: boolean;
}

const DecisionModelGarden: React.FC<DecisionModelGardenProps> = ({ canAddModels }) => {
  const { data: costMap } = useModelCostMap(true, true);
  const { data: providerFields } = useProviderFields();
  const rows = useMemo(() => buildDecisionGarden(costMap), [costMap]);
  const addableProviders = useMemo(
    () => new Set((providerFields ?? []).map((info) => info.litellm_provider)),
    [providerFields],
  );
  const [showAll, setShowAll] = useState(false);

  if (rows.length === 0) {
    return null;
  }

  const routedRows = rows.filter((row) => row.route === "decisions");
  const routedProviderCount = new Set(routedRows.map((row) => row.provider)).size;
  const selfHostedCount = rows.length - routedRows.length;
  const visibleRows = showAll ? rows : rows.slice(0, PREVIEW_COUNT);

  return (
    <Card className="mt-6 px-6" data-testid="decision-model-garden">
      <div className="mb-4 flex items-center gap-4">
        <div className="flex size-10 shrink-0 items-center justify-center rounded-full border bg-background">
          <Sparkles className="size-4 text-muted-foreground" />
        </div>
        <div className="min-w-0 flex-1">
          <h3 className="m-0 text-base font-semibold text-foreground">Decision models LiteLLM supports</h3>
          <p className="m-0 mt-0.5 text-sm text-muted-foreground">
            {routedRows.length} decision models across {routedProviderCount} providers answer on{" "}
            <code>/v1/decisions</code> once added to this proxy.
            {selfHostedCount > 0 && ` ${selfHostedCount} more are self-hosted and answer on their own routes.`}
          </p>
        </div>
        <Button
          variant="outline"
          className="shrink-0"
          nativeButton={false}
          role="link"
          render={<a href={DECISIONS_DOCS_URL} target="_blank" rel="noopener noreferrer" />}
        >
          <BookOpen />
          Decisions docs
        </Button>
      </div>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {visibleRows.map((row) => (
          <DecisionModelCard
            key={row.model}
            row={row}
            canAddModels={canAddModels && row.route === "decisions" && addableProviders.has(row.provider)}
          />
        ))}
      </div>
      {rows.length > PREVIEW_COUNT && (
        <div className="mt-4 flex justify-center">
          <Button variant="ghost" type="button" onClick={() => setShowAll((value) => !value)}>
            {showAll ? "Show fewer" : `Show all ${rows.length}`}
          </Button>
        </div>
      )}
    </Card>
  );
};

export default DecisionModelGarden;
