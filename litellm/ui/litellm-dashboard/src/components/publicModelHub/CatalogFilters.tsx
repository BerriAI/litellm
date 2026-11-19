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
      className="border-r border-r-border pr-[26px] max-[1099px]:border max-[1099px]:border-border max-[1099px]:pr-0"
    >
      <Collapsible open={!narrow || expanded} onOpenChange={setExpanded}>
        <div
          data-slot="catalog-filter-heading"
          className="flex items-center justify-between gap-2 pb-5 max-[1099px]:px-4 max-[1099px]:py-3"
        >
          <h2 className="text-sm leading-[1.6] font-semibold max-[1099px]:hidden">
            {t("publicModelHub.catalog.filters")}
          </h2>
          <CollapsibleTrigger
            render={
              <Button
                variant="ghost"
                className="hidden h-auto flex-1 cursor-pointer items-center justify-between gap-2 p-0 text-sm leading-[1.6] font-semibold focus-visible:outline-2 focus-visible:outline-offset-4 max-[1099px]:flex [&_svg]:size-3.5"
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
        <CollapsibleContent className="flex flex-col gap-[22px] max-[1099px]:grid max-[1099px]:grid-cols-2 max-[1099px]:gap-5 max-[1099px]:px-4 max-[1099px]:pt-1 max-[1099px]:pb-[18px]">
          {children}
        </CollapsibleContent>
      </Collapsible>
    </aside>
  );
}
