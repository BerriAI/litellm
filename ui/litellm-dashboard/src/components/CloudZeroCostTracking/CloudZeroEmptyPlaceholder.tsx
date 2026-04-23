import { Button } from "@/components/ui/button";
import { Inbox } from "lucide-react";

interface CloudZeroEmptyPlaceholderProps {
  startCreation: () => void;
}

export default function CloudZeroEmptyPlaceholder({
  startCreation,
}: CloudZeroEmptyPlaceholderProps) {
  return (
    <div className="bg-background p-12 rounded-lg border border-dashed border-border text-center max-w-2xl mx-auto mt-8">
      <div className="flex flex-col items-center gap-3">
        <Inbox className="h-12 w-12 text-muted-foreground" />
        <h4 className="text-lg font-semibold">
          No CloudZero Integration Found
        </h4>
        <p className="text-sm text-muted-foreground max-w-md mx-auto">
          Connect your CloudZero account to start tracking and analyzing your
          cloud costs directly from LiteLLM.
        </p>
        <Button onClick={startCreation} className="mt-4">
          Add CloudZero Integration
        </Button>
      </div>
    </div>
  );
}
