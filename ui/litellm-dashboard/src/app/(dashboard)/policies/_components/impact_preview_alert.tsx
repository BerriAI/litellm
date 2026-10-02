import React from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { Badge } from "@/components/ui/badge";
import { AlertTriangle, Info } from "lucide-react";

interface ImpactResult {
  affected_keys_count: number;
  affected_teams_count: number;
  sample_keys: string[];
  sample_teams: string[];
}

interface ImpactPreviewAlertProps {
  impactResult: ImpactResult;
  isDefault?: boolean;
}

interface SampleListProps {
  label: string;
  samples: string[];
  totalCount: number;
}

const SampleList: React.FC<SampleListProps> = ({ label, samples, totalCount }) => (
  <div className="mt-1 flex flex-wrap items-center gap-1">
    <span className="text-xs text-muted-foreground">{label}: </span>
    {samples.slice(0, 5).map((sample) => (
      <Badge key={sample} variant="outline">
        {sample}
      </Badge>
    ))}
    {totalCount > 5 && <span className="text-xs text-muted-foreground">and {totalCount - 5} more...</span>}
  </div>
);

const ImpactPreviewAlert: React.FC<ImpactPreviewAlertProps> = ({ impactResult, isDefault = false }) => {
  const isGlobal = impactResult.affected_keys_count === -1;
  const qualifier = isDefault ? "up to " : "";

  return (
    <Alert className="mb-4">
      {isGlobal ? <AlertTriangle /> : <Info />}
      <AlertTitle>Impact Preview</AlertTitle>
      <AlertDescription>
        {isGlobal ? (
          <span>
            Global scope — this will affect <strong>all keys and teams</strong>.
          </span>
        ) : (
          <div>
            <span>
              This attachment would affect {qualifier}
              <strong>
                {impactResult.affected_keys_count} key{impactResult.affected_keys_count !== 1 ? "s" : ""}
              </strong>{" "}
              and{" "}
              <strong>
                {impactResult.affected_teams_count} team{impactResult.affected_teams_count !== 1 ? "s" : ""}
              </strong>
              .
            </span>
            {isDefault && (
              <div className="text-xs text-muted-foreground">
                Default attachments only apply to requests no non-default attachment matches, so fewer may be affected.
              </div>
            )}
            {impactResult.sample_keys.length > 0 && (
              <SampleList
                label="Keys"
                samples={impactResult.sample_keys}
                totalCount={impactResult.affected_keys_count}
              />
            )}
            {impactResult.sample_teams.length > 0 && (
              <SampleList
                label="Teams"
                samples={impactResult.sample_teams}
                totalCount={impactResult.affected_teams_count}
              />
            )}
          </div>
        )}
      </AlertDescription>
    </Alert>
  );
};

export default ImpactPreviewAlert;
