import { Button } from "@/components/ui/button";
import { Inbox } from "lucide-react";

interface HashicorpVaultEmptyPlaceholderProps {
  onAdd: () => void;
}

export default function HashicorpVaultEmptyPlaceholder({
  onAdd,
}: HashicorpVaultEmptyPlaceholderProps) {
  return (
    <div className="bg-background p-12 rounded-lg border border-dashed border-border text-center w-full">
      <div className="flex flex-col items-center gap-3">
        <Inbox className="h-12 w-12 text-muted-foreground" />
        <h4 className="text-lg font-semibold">
          No Vault Configuration Found
        </h4>
        <p className="text-sm text-muted-foreground max-w-md mx-auto">
          Configure Hashicorp Vault to securely manage provider API keys and
          secrets for your LiteLLM deployment.
        </p>
        <Button onClick={onAdd} className="mt-4">
          Configure Vault
        </Button>
      </div>
    </div>
  );
}
