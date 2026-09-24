import { KeyRound } from "lucide-react";

import { Button } from "@/components/ui/button";

interface CyberArkEmptyPlaceholderProps {
  onAdd: () => void;
}

export default function CyberArkEmptyPlaceholder({ onAdd }: CyberArkEmptyPlaceholderProps) {
  return (
    <div className="flex w-full flex-col items-center rounded-lg border border-dashed border-border bg-card p-12 text-center">
      <div className="mb-4 flex size-12 items-center justify-center rounded-full bg-muted">
        <KeyRound className="size-6 text-muted-foreground" />
      </div>
      <h4 className="text-base font-semibold text-foreground">No CyberArk Configuration Found</h4>
      <p className="mx-auto mt-2 max-w-md text-sm text-muted-foreground">
        Configure CyberArk Conjur to securely manage provider API keys and secrets for your LiteLLM deployment.
      </p>
      <Button size="lg" onClick={onAdd} className="mt-4">
        Configure CyberArk
      </Button>
    </div>
  );
}
