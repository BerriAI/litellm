"use client";

import { PanelRightClose } from "lucide-react";

import { Button } from "@/components/ui/button";

import { IdChip } from "../../ui/IdChip";
import { SpanIcon } from "../../ui/SpanIcon";
import type { SpanType } from "../../types";

interface PaneHeaderProps {
  type: SpanType;
  model: string | null;
  failed: boolean;
  title: React.ReactNode;
  idValue?: string;
  facts: readonly (readonly [label: string, value: string])[];
  actions?: React.ReactNode;
  links?: React.ReactNode;
  onClose: () => void;
}

/** Selected step identity: tile, name, id, then its time and usage on one quiet line. */
export function PaneHeader({ type, model, failed, title, idValue, facts, actions, links, onClose }: PaneHeaderProps) {
  return (
    <div className="flex shrink-0 flex-col gap-1.5 px-4 pt-3 pb-2">
      <div className="flex min-w-0 items-center gap-2">
        <SpanIcon type={type} model={model} error={failed} size="lg" />
        <h2 className="min-w-0 truncate text-base font-semibold text-foreground">{title}</h2>
        {idValue && <IdChip value={idValue} label="Copy span ID" />}
        <div className="ml-auto flex shrink-0 items-center gap-1">
          {actions}
          <Button
            variant="ghost"
            size="icon-xs"
            onClick={onClose}
            aria-label="Close details"
            className="text-muted-foreground"
          >
            <PanelRightClose className="size-4" />
          </Button>
        </div>
      </div>
      {(facts.length > 0 || links) && (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 pl-8">
          <dl className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground tabular-nums">
            {facts.map(([label, value]) => (
              <div key={label} className="flex items-center gap-1">
                <dt>{label}</dt>
                <dd className="font-medium text-foreground">{value}</dd>
              </div>
            ))}
          </dl>
          {links}
        </div>
      )}
    </div>
  );
}
