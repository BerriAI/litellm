"use client";

import { ChevronDown } from "lucide-react";
import { useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";

import { useIsNarrow } from "./useIsNarrow";
import styles from "./catalog.module.css";

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
        <div className={`${styles.filterHeading} flex items-center justify-between gap-2 pb-5`}>
          <h2 className={`${styles.filterTitle} text-sm leading-[1.6] font-semibold`}>
            {t("publicModelHub.catalog.filters")}
          </h2>
          <CollapsibleTrigger
            render={<button type="button" className={`${styles.filterTrigger} text-sm leading-[1.6] font-semibold`} />}
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
        <CollapsibleContent className={`${styles.filterContent} flex flex-col gap-[22px]`}>
          {children}
        </CollapsibleContent>
      </Collapsible>
    </aside>
  );
}
