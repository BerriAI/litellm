"use client";

import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";

export function InvestigationNavigation({ onBack }: { onBack: () => void }) {
  return (
    <header>
      <Button variant="ghost" size="sm" className="-ml-3" onClick={onBack}>
        <ArrowLeft className="size-4" /> Back
      </Button>
    </header>
  );
}
