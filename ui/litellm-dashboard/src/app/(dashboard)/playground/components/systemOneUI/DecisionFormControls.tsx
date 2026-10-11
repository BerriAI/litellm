import { Button } from "@/components/ui/button";
import { Plus, Trash2 } from "lucide-react";
import { useId, type ReactNode } from "react";
import { IssueList } from "./ValidationFeedback";
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

export function QuestionCardHeader({
  index,
  removeLabel,
  onRemove,
}: {
  index: number;
  removeLabel: string;
  onRemove: () => void;
}) {
  return (
    <div className="flex items-center justify-between gap-2">
      <span className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">Question {index + 1}</span>
      <RemoveButton label={removeLabel} onClick={onRemove} />
    </div>
  );
}

export function FormFrame({ validation, children }: { validation: PayloadValidation<unknown>; children: ReactNode }) {
  const issuesId = useId();

  return (
    <div className="flex min-h-96 flex-1 flex-col overflow-hidden">
      <div className="grid min-h-80 flex-1 content-start gap-5 overflow-auto p-4">{children}</div>
      <IssueList id={issuesId} validation={validation} />
    </div>
  );
}
