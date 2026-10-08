"use client";

import { useId, useState, type FormEvent } from "react";
import { Loader2, Pencil } from "lucide-react";
import type { KeyResponse } from "@/components/key_team_helpers/key_list";
import BudgetDurationDropdown from "@/components/common_components/budget_duration_dropdown";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Popover, PopoverContent, PopoverTitle, PopoverTrigger } from "@/components/ui/popover";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { canonicalBudgetDuration } from "@/components/templates/keyEditFieldNormalizers";
import { budgetQuickEditPayload } from "./quickEditPayload";
import { useQuickKeyUpdate } from "./useQuickKeyUpdate";

interface KeyBudgetQuickEditProps {
  keyData: KeyResponse;
  accessToken: string | null;
  canModify: boolean;
  onKeyDataUpdate?: (updated: Partial<KeyResponse>) => void;
  buttonClassName?: string;
  buttonVariant?: "ghost" | "outline";
  showEditLabel?: boolean;
}

export function KeyBudgetQuickEdit({
  keyData,
  accessToken,
  canModify,
  onKeyDataUpdate,
  buttonClassName,
  buttonVariant = "ghost",
  showEditLabel = false,
}: KeyBudgetQuickEditProps) {
  const inputId = useId();
  const durationId = useId();
  const [open, setOpen] = useState(false);
  const [maxBudget, setMaxBudget] = useState("");
  const [budgetDuration, setBudgetDuration] = useState<string | null>(null);
  const [validationError, setValidationError] = useState("");
  const mutation = useQuickKeyUpdate(accessToken ?? "");

  const isAllowed = canModify && Boolean(accessToken);
  const handleOpenChange = (nextOpen: boolean) => {
    setOpen(nextOpen);
    if (!nextOpen) return;

    setMaxBudget(keyData.max_budget == null ? "" : String(keyData.max_budget));
    setBudgetDuration(canonicalBudgetDuration(keyData.budget_duration));
    setValidationError("");
  };

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (mutation.isPending) return;

    const result = budgetQuickEditPayload(keyData.token || keyData.token_id, {
      maxBudget,
      budgetDuration,
      currentBudgetDuration: keyData.budget_duration,
    });
    if (result.kind === "invalid") {
      setValidationError(result.message);
      return;
    }

    setValidationError("");
    mutation.mutate(
      { kind: "budget", payload: result.payload },
      {
        onSuccess: (updated) => {
          onKeyDataUpdate?.(updated);
          setOpen(false);
        },
      },
    );
  };

  if (!isAllowed) return null;

  return (
    <Popover open={open} onOpenChange={handleOpenChange}>
      <PopoverTrigger
        render={
          <Button
            type="button"
            variant={buttonVariant}
            size={showEditLabel ? "sm" : "icon-xs"}
            aria-label="Edit budget"
            className={`${showEditLabel ? "" : "text-muted-foreground opacity-50 transition-opacity group-hover/editable:opacity-100 focus-visible:opacity-100"} ${buttonClassName ?? ""}`}
          >
            <Pencil aria-hidden="true" />
            {showEditLabel && "Edit"}
          </Button>
        }
      />
      <PopoverContent align="end" className="gap-3">
        <PopoverTitle>Quick edit budget</PopoverTitle>
        <form
          className="flex flex-col gap-3"
          onSubmit={handleSubmit}
          onKeyDown={(event) => {
            if (event.key === "Escape") setOpen(false);
          }}
        >
          <div className="flex flex-col gap-1.5">
            <label htmlFor={inputId} className="text-sm font-medium">
              Max budget (USD)
            </label>
            <Input
              id={inputId}
              type="number"
              min="0"
              step="any"
              autoFocus
              placeholder="Unlimited"
              value={maxBudget}
              onChange={(event) => setMaxBudget(event.target.value)}
              aria-invalid={validationError !== ""}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <label htmlFor={durationId} className="text-sm font-medium">
              Resets
            </label>
            <BudgetDurationDropdown
              id={durationId}
              value={budgetDuration}
              onChange={setBudgetDuration}
              placeholder="Never resets"
            />
          </div>
          <p className="text-xs text-muted-foreground">Current spend: ${formatNumberWithCommas(keyData.spend, 4)}</p>
          {validationError && (
            <p role="alert" className="text-xs text-destructive">
              {validationError}
            </p>
          )}
          <div className="flex justify-end gap-2">
            <Button type="button" variant="outline" size="sm" onClick={() => setOpen(false)}>
              Cancel
            </Button>
            <Button type="submit" size="sm" disabled={mutation.isPending}>
              {mutation.isPending && <Loader2 className="size-4 animate-spin" aria-hidden="true" />}
              Save
            </Button>
          </div>
        </form>
      </PopoverContent>
    </Popover>
  );
}
