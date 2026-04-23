import React from "react";
import { Badge } from "@/components/ui/badge";
import { AlertTriangle, Info } from "lucide-react";
import { cn } from "@/lib/utils";

interface ImpactResult {
  affected_keys_count: number;
  affected_teams_count: number;
  sample_keys: string[];
  sample_teams: string[];
}

interface ImpactPreviewAlertProps {
  impactResult: ImpactResult;
}

const ImpactPreviewAlert: React.FC<ImpactPreviewAlertProps> = ({
  impactResult,
}) => {
  const isWarning = impactResult.affected_keys_count === -1;

  return (
    <div
      className={cn(
        "mb-4 rounded-md border p-3 flex gap-2 items-start",
        isWarning
          ? "bg-amber-50 border-amber-200 text-amber-900 dark:bg-amber-950/30 dark:border-amber-900 dark:text-amber-200"
          : "bg-blue-50 border-blue-200 text-blue-900 dark:bg-blue-950/30 dark:border-blue-900 dark:text-blue-200",
      )}
    >
      {isWarning ? (
        <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0" />
      ) : (
        <Info className="h-4 w-4 mt-0.5 shrink-0" />
      )}
      <div className="flex-1">
        <div className="font-semibold">Impact Preview</div>
        {isWarning ? (
          <div className="text-sm mt-1">
            Global scope — this will affect{" "}
            <strong>all keys and teams</strong>.
          </div>
        ) : (
          <div className="text-sm mt-1">
            <div>
              This attachment would affect{" "}
              <strong>
                {impactResult.affected_keys_count} key
                {impactResult.affected_keys_count !== 1 ? "s" : ""}
              </strong>{" "}
              and{" "}
              <strong>
                {impactResult.affected_teams_count} team
                {impactResult.affected_teams_count !== 1 ? "s" : ""}
              </strong>
              .
            </div>
            {impactResult.sample_keys.length > 0 && (
              <div className="mt-1">
                <span className="text-xs text-muted-foreground">Keys: </span>
                {impactResult.sample_keys.slice(0, 5).map((k: string) => (
                  <Badge
                    key={k}
                    variant="outline"
                    className="text-[11px] mr-1"
                  >
                    {k}
                  </Badge>
                ))}
                {impactResult.affected_keys_count > 5 && (
                  <span className="text-[11px] text-muted-foreground">
                    and {impactResult.affected_keys_count - 5} more...
                  </span>
                )}
              </div>
            )}
            {impactResult.sample_teams.length > 0 && (
              <div className="mt-1">
                <span className="text-xs text-muted-foreground">Teams: </span>
                {impactResult.sample_teams.slice(0, 5).map((t: string) => (
                  <Badge
                    key={t}
                    variant="outline"
                    className="text-[11px] mr-1"
                  >
                    {t}
                  </Badge>
                ))}
                {impactResult.affected_teams_count > 5 && (
                  <span className="text-[11px] text-muted-foreground">
                    and {impactResult.affected_teams_count - 5} more...
                  </span>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ImpactPreviewAlert;
