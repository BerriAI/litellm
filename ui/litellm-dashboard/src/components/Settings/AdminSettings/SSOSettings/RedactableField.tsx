import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Eye, EyeOff } from "lucide-react";

export default function RedactableField({
  defaultHidden = true,
  value,
}: {
  defaultHidden?: boolean;
  value: string | null;
}) {
  const [isHidden, setIsHidden] = useState(defaultHidden);

  return (
    <div className="flex items-center gap-2">
      <span className="font-mono text-muted-foreground flex-1">
        {value ? (
          isHidden ? (
            "•".repeat(value.length)
          ) : (
            value
          )
        ) : (
          <span className="text-muted-foreground italic">Not configured</span>
        )}
      </span>
      {value && (
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7"
          onClick={() => setIsHidden(!isHidden)}
          aria-label={isHidden ? "Show value" : "Hide value"}
        >
          {isHidden ? (
            <Eye className="w-4 h-4" />
          ) : (
            <EyeOff className="w-4 h-4" />
          )}
        </Button>
      )}
    </div>
  );
}
