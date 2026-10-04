"use client";

import { ArrowLeft, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";

export function InvestigationNavigation({
  showActions,
  ready,
  onBack,
  onCreate,
}: {
  showActions: boolean;
  ready: boolean;
  onBack: () => void;
  onCreate: () => void;
}) {
  return (
    <header className="flex flex-wrap items-center justify-between gap-2">
      <Button variant="ghost" size="sm" className="-ml-3" onClick={onBack}>
        <ArrowLeft className="size-4" /> Back
      </Button>
      {showActions && (
        <Button size="sm" variant="outline" disabled={!ready} onClick={onCreate}>
          <Plus className="size-4" /> New investigation
        </Button>
      )}
    </header>
  );
}
