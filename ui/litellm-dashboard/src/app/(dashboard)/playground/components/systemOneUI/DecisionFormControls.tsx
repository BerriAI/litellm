import { Button } from "@/components/ui/button";
import { cn } from "@/lib/cva.config";
import { Plus, Trash2 } from "lucide-react";
import { useId, type ReactNode } from "react";
import { IssueList, ValidationStatus } from "./JsonEditor";
import type { PayloadValidation } from "./lib/validatePayload";

export function RemoveButton({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <Button variant="ghost" size="icon" aria-label={label} title={label} onClick={onClick}>
      <Trash2 />
    </Button>
  );
}

export function AddButton({ children, onClick }: { children: ReactNode; onClick: () => void }) {
  return (
    <Button variant="outline" size="sm" className="w-fit" onClick={onClick}>
      <Plus />
      {children}
    </Button>
  );
}

export function FormFrame({ validation, children }: { validation: PayloadValidation<unknown>; children: ReactNode }) {
  const issuesId = useId();

  return (
    <div
      className={cn(
        "flex min-h-96 flex-1 flex-col overflow-hidden rounded-md border bg-background",
        !validation.isValid && "border-destructive/60",
      )}
    >
      <div className="flex items-center gap-2 border-b px-3 py-2">
        <span className="text-sm font-medium">Request form</span>
        <ValidationStatus validation={validation} />
      </div>
      <div className="grid min-h-80 flex-1 content-start gap-4 overflow-auto p-3">{children}</div>
      <IssueList id={issuesId} validation={validation} />
    </div>
  );
}
