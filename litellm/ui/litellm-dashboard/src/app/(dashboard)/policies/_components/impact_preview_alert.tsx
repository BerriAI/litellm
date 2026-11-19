import React from "react";
import { Trans, useTranslation } from "react-i18next";
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
  const { t } = useTranslation();
  return (
    <div className="mt-1 flex flex-wrap items-center gap-1">
      <span className="text-xs text-muted-foreground">{label}: </span>
      {samples.slice(0, 5).map((sample) => (
        <Badge key={sample} variant="outline">
          {sample}
        </Badge>
      ))}
      {totalCount > 5 && (
        <span className="text-xs text-muted-foreground">
          {t("policies.impactPreview.more", { remaining: totalCount - 5 })}
        </span>
      )}
    </div>
  );
};

const ImpactPreviewAlert: React.FC<ImpactPreviewAlertProps> = ({ impactResult, isDefault = false }) => {
  const { t } = useTranslation();
  const isGlobal = impactResult.affected_keys_count === -1;

  return (
    <Alert className="mb-4">
      {isGlobal ? <AlertTriangle /> : <Info />}
      <AlertTitle>{t("policies.impactPreview.title")}</AlertTitle>
      <AlertDescription>
        {isGlobal ? (
          <span>
            <Trans i18nKey="policies.impactPreview.global" components={{ strong: <strong /> }} />
          </span>
        ) : (
          <div>
            <span>
              <Trans
                i18nKey={isDefault ? "policies.impactPreview.summaryUpTo" : "policies.impactPreview.summary"}
                values={{ keys: impactResult.affected_keys_count, teams: impactResult.affected_teams_count }}
                components={{ strong: <strong /> }}
              />
            </span>
            {isDefault && (
              <div className="text-xs text-muted-foreground">{t("policies.impactPreview.defaultNote")}</div>
            )}
            {impactResult.sample_keys.length > 0 && (
              <SampleList
                label={t("policies.impactPreview.sampleKeys")}
                samples={impactResult.sample_keys}
                totalCount={impactResult.affected_keys_count}
              />
            )}
            {impactResult.sample_teams.length > 0 && (
              <SampleList
                label={t("policies.impactPreview.sampleTeams")}
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
