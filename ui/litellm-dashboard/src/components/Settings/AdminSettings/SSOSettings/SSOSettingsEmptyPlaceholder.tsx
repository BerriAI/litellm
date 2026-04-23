import { Button } from "@/components/ui/button";
import { Inbox } from "lucide-react";

interface SSOSettingsEmptyPlaceholderProps {
  onAdd: () => void;
}

export default function SSOSettingsEmptyPlaceholder({
  onAdd,
}: SSOSettingsEmptyPlaceholderProps) {
  return (
    <div className="bg-background p-12 rounded-lg border border-dashed border-border text-center w-full">
      <div className="flex flex-col items-center gap-3">
        <Inbox className="h-12 w-12 text-muted-foreground" />
        <h4 className="text-lg font-semibold">No SSO Configuration Found</h4>
        <p className="text-sm text-muted-foreground max-w-md mx-auto">
          Configure Single Sign-On (SSO) to enable seamless authentication for
          your team members using your identity provider.
        </p>
        <Button onClick={onAdd} className="mt-4">
          Configure SSO
        </Button>
      </div>
    </div>
  );
}
