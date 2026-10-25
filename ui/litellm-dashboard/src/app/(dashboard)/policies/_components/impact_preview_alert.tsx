import { useLanguage } from "@/contexts/LanguageContext";
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

const SampleList: React.FC<SampleListProps> = ({ label, samples, totalCount }) => {
  const { t } = useLanguage();
  return (
    <div className="mt-1 flex flex-wrap items-center gap-1">
      <span className="text-xs text-muted-foreground">{label}: </span>
      {samples.slice(0, 5).map((sample) => (
        <Badge key={sample} variant="outline">
          {sample}
        </Badge>
      ))}
      {totalCount > 5 && (
        <span className="text-xs text-muted-foreground">{t("and {count} more...", { count: totalCount - 5 })}</span>
      )}
    </div>
  );
};

const ImpactPreviewAlert: React.FC<ImpactPreviewAlertProps> = ({ impactResult, isDefault = false }) => {
  const { t } = useLanguage();
  const isGlobal = impactResult.affected_keys_count === -1;
  const counts = {
    keys: t(impactResult.affected_keys_count === 1 ? "{count} key" : "{count} keys", {
      count: impactResult.affected_keys_count,
    }),
    teams: t(impactResult.affected_teams_count === 1 ? "{count} team" : "{count} teams", {
      count: impactResult.affected_teams_count,
    }),
  };

  return (
    <Alert className="mb-4">
      {isGlobal ? <AlertTriangle /> : <Info />}
      <AlertTitle>{t("Impact Preview")}</AlertTitle>
      <AlertDescription>
        {isGlobal ? (
          <span>{t("Global scope — this will affect all keys and teams.")}</span>
        ) : (
          <div>
            <span>
              {t(
                isDefault
                  ? "This attachment would affect up to {keys} and {teams}."
                  : "This attachment would affect {keys} and {teams}.",
                counts,
              )}
            </span>
            {isDefault && (
              <div className="text-xs text-muted-foreground">
                {t(
                  "Default attachments only apply to requests no non-default attachment matches, so fewer may be affected.",
                )}
              </div>
            )}
            {impactResult.sample_keys.length > 0 && (
              <SampleList
                label={t("Keys")}
                samples={impactResult.sample_keys}
                totalCount={impactResult.affected_keys_count}
              />
            )}
            {impactResult.sample_teams.length > 0 && (
              <SampleList
                label={t("Teams")}
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
