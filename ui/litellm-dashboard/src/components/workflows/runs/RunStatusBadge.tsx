import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/cva.config";
import type { RunState } from "./contracts";
import { stateLabels } from "./presentation";

const stateColors: Record<RunState, string> = {
  pending: "bg-muted text-muted-foreground",
  running: "bg-info/10 text-info",
  completed: "bg-success/10 text-success",
  failed: "bg-destructive/10 text-destructive",
  cancelled: "bg-muted text-muted-foreground",
};

export function RunStatusBadge({ state }: { state: RunState }) {
  return (
    <Badge variant="outline" className={cn("gap-1.5", stateColors[state])}>
      <span aria-hidden className="size-1.5 rounded-full bg-current" />
      {stateLabels[state]}
    </Badge>
  );
}
