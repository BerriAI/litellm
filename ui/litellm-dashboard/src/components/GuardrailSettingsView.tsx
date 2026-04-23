import React from "react";
import { Badge } from "@/components/ui/badge";
import { Globe } from "lucide-react";
import { cn } from "@/lib/utils";

interface GuardrailSettingsViewProps {
  globalGuardrailNames: Set<string>;
  teamGuardrails?: string[];
  optedOutGlobalGuardrails?: string[];
  killSwitchOn?: boolean;
  variant?: "card" | "inline";
  className?: string;
}

export function GuardrailSettingsView({
  globalGuardrailNames,
  teamGuardrails = [],
  optedOutGlobalGuardrails = [],
  killSwitchOn = false,
  variant = "card",
  className = "",
}: GuardrailSettingsViewProps) {
  const optedOutSet = new Set(optedOutGlobalGuardrails);
  const globalsRunning = Array.from(globalGuardrailNames).filter(
    (n) => !optedOutSet.has(n),
  );
  const nonGlobalOptIns = teamGuardrails.filter(
    (n) => !globalGuardrailNames.has(n),
  );

  const isEmpty =
    !killSwitchOn &&
    globalsRunning.length === 0 &&
    nonGlobalOptIns.length === 0;

  const content = isEmpty ? (
    <span className="block text-muted-foreground">No guardrails configured</span>
  ) : (
    <div className="flex flex-col gap-4">
      <div>
        <span className="block text-sm font-medium text-foreground mb-2">
          <Globe
            className="inline-block h-4 w-4 mr-1"
            aria-label="Global guardrail"
          />
          Global
        </span>
        {killSwitchOn ? (
          <Badge className="bg-amber-500 text-white hover:bg-amber-500">
            Bypassed for this team
          </Badge>
        ) : globalsRunning.length > 0 ? (
          <div className="flex flex-wrap gap-2">
            {globalsRunning.map((name) => (
              <Badge key={name} variant="default">
                {name}
              </Badge>
            ))}
          </div>
        ) : (
          <span className="block text-sm text-muted-foreground">
            None configured
          </span>
        )}
      </div>
      <div>
        <span className="block text-sm font-medium text-foreground mb-2">
          Team-specific
        </span>
        {nonGlobalOptIns.length > 0 ? (
          <div className="flex flex-wrap gap-2">
            {nonGlobalOptIns.map((name) => (
              <Badge key={name} variant="default">
                {name}
              </Badge>
            ))}
          </div>
        ) : (
          <span className="block text-sm text-muted-foreground">
            None configured
          </span>
        )}
      </div>
    </div>
  );

  if (variant === "card") {
    return (
      <div
        className={cn(
          "bg-background border border-border rounded-lg p-6",
          className,
        )}
      >
        <div className="flex items-center gap-2 mb-6">
          <div>
            <span className="block font-semibold text-foreground">
              Guardrails Settings
            </span>
            <span className="block text-xs text-muted-foreground">
              Global and team-specific guardrails applied to this team
            </span>
          </div>
        </div>
        {content}
      </div>
    );
  }

  return (
    <div className={className}>
      <span className="block font-medium text-foreground mb-3">
        Guardrails Settings
      </span>
      {content}
    </div>
  );
}

export default GuardrailSettingsView;
