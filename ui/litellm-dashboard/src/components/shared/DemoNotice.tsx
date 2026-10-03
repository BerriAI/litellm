import { Info } from "lucide-react";
import { Button } from "@/components/ui/button";

export function DemoNotice({ onExit }: { onExit?: () => void }) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-info/20 bg-info/10 px-4 py-3">
      <p role="status" className="flex items-center gap-2 text-sm font-medium text-info">
        <Info aria-hidden="true" className="size-4 shrink-0" />
        You’re viewing demo data
      </p>
      <Button variant="outline" size="sm" onClick={onExit}>
        Exit demo
      </Button>
    </div>
  );
}
