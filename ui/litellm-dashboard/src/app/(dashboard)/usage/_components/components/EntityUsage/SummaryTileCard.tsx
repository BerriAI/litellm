import { ChevronDown, ChevronRight, Info } from "lucide-react";

import { Card as ShadcnCard, CardContent } from "@/components/ui/card";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

import type { SummaryTile } from "./entityUsageSummary";

interface SummaryTileCardProps {
  tile: SummaryTile;
  expanded?: boolean;
  onToggleExpand?: () => void;
}

export function SummaryTileCard({ tile, expanded = false, onToggleExpand }: SummaryTileCardProps) {
  const { title, value, className, tooltip, expandable } = tile;
  const chev = "size-3 text-muted-foreground";
  const expandIcon = expanded ? <ChevronDown className={chev} /> : <ChevronRight className={chev} />;

  return (
    <ShadcnCard
      className={expandable ? "cursor-pointer hover:bg-accent transition-colors" : undefined}
      onClick={expandable ? onToggleExpand : undefined}
    >
      <CardContent>
        <div className="flex items-center gap-2">
          <h3 className="text-lg font-medium text-foreground">{title}</h3>
          {tooltip ? (
            <Tooltip>
              <TooltipTrigger render={<Info className="size-4 text-muted-foreground hover:text-foreground" />} />
              <TooltipContent>{tooltip}</TooltipContent>
            </Tooltip>
          ) : null}
          {expandable ? expandIcon : null}
        </div>
        <p className={`text-2xl font-bold mt-2 ${className ?? ""}`}>{value}</p>
      </CardContent>
    </ShadcnCard>
  );
}

export default SummaryTileCard;
