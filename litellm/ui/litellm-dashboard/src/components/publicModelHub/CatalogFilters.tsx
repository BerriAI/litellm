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
    <aside
      data-slot="catalog-filters"
      className="border-r border-r-border pr-[26px] catalog-narrow:border catalog-narrow:border-border catalog-narrow:pr-0"
    >
      <Collapsible open={!narrow || expanded} onOpenChange={setExpanded}>
        <div
          data-slot="catalog-filter-heading"
          className="flex items-center justify-between gap-2 pb-5 catalog-narrow:px-4 catalog-narrow:py-3"
        >
          <h2 className="text-sm leading-[1.6] font-semibold catalog-narrow:hidden">
            {t("publicModelHub.catalog.filters")}
          </h2>
          <CollapsibleTrigger
            aria-expanded={expanded}
            render={
              <Button
                variant="ghost"
                className="hidden h-auto flex-1 cursor-pointer items-center justify-between gap-2 p-0 text-sm leading-[1.6] font-semibold focus-visible:outline-2 focus-visible:outline-offset-4 catalog-narrow:flex [&_svg]:size-3.5"
              />
            }
          >
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
        <CollapsibleContent
          data-expanded={expanded}
          className="catalog-narrow:data-[expanded=false]:hidden flex flex-col gap-[22px] catalog-narrow:grid catalog-narrow:grid-cols-[repeat(auto-fit,minmax(min(100%,12rem),1fr))] catalog-narrow:gap-5 catalog-narrow:px-4 catalog-narrow:pt-1 catalog-narrow:pb-[18px]"
        >
          {children}
        </CollapsibleContent>
      </Collapsible>
    </aside>
  );
}
