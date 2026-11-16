"use client";

import { ChevronDown } from "lucide-react";
import { useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";

import { useIsNarrow } from "./useIsNarrow";

export function CatalogFilters({
  children,
  count,
  onClear,
}: {
  children: ReactNode;
  count: number;
  onClear: () => void;
}) {
  const { t } = useTranslation();
  const narrow = useIsNarrow();
  const [expanded, setExpanded] = useState(false);
  return (
    <aside data-slot="catalog-filters">
      <Collapsible open={!narrow || expanded} onOpenChange={setExpanded}>
        <div className="mb-5 flex items-center justify-between gap-2">
          <h2 className="hidden text-sm font-semibold lg:block">{t("publicModelHub.catalog.filters")}</h2>
          <CollapsibleTrigger render={<Button variant="outline" className="lg:hidden" />}>
            {t("publicModelHub.catalog.filters")}
            {count > 0 && <span className="tabular-nums">({count})</span>}
            <ChevronDown />
          </CollapsibleTrigger>
          {count > 0 && (
            <Button variant="ghost" size="sm" onClick={onClear}>
              {t("publicModelHub.catalog.clearFilters")}
            </Button>
          )}
        </div>
        <CollapsibleContent className="flex flex-col gap-5">{children}</CollapsibleContent>
      </Collapsible>
    </aside>
  );
}
