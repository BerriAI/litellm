import { Button } from "@/components/ui/button";
import { Inbox } from "lucide-react";

interface CloudZeroEmptyPlaceholderProps {
  startCreation: () => void;
}

export default function CloudZeroEmptyPlaceholder({ startCreation }: CloudZeroEmptyPlaceholderProps) {
  return (
    <div className="mx-auto mt-8 max-w-2xl rounded-lg border border-dashed border-border bg-card p-12 text-center">
      <div className="flex flex-col items-center gap-2">
        <Inbox className="size-10 text-muted-foreground" aria-hidden />
        <h4 className="text-base font-semibold">No CloudZero Integration Found</h4>
        <p className="mx-auto max-w-md text-sm text-muted-foreground">
          Connect your CloudZero account to start tracking and analyzing your cloud costs directly from LiteLLM.
        </p>
        <Button size="lg" onClick={startCreation} className="mt-4">
          Add CloudZero Integration
        </Button>
      </div>
    </div>
  );
}
