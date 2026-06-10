import React from "react";
import { Alert, Tag, Typography } from "antd";
import { useTranslation, Trans } from "react-i18next";

const { Text } = Typography;

interface ImpactResult {
  affected_keys_count: number;
  affected_teams_count: number;
  sample_keys: string[];
  sample_teams: string[];
}

interface ImpactPreviewAlertProps {
  impactResult: ImpactResult;
}

const ImpactPreviewAlert: React.FC<ImpactPreviewAlertProps> = ({ impactResult }) => {
  const { t } = useTranslation();

  return (
    <Alert
      type={impactResult.affected_keys_count === -1 ? "warning" : "info"}
      showIcon
      className="mb-4"
      message={t("policies.impactPreviewAlert.message")}
      description={
        impactResult.affected_keys_count === -1 ? (
          <Text>
            <Trans i18nKey="policies.impactPreviewAlert.globalScopeDesc" components={{ strong: <strong /> }} />
          </Text>
        ) : (
          <div>
            <Text>
              <Trans
                i18nKey="policies.impactPreviewAlert.attachmentAffects"
                values={{
                  keysCount: impactResult.affected_keys_count,
                  keysWord: t("policies.impactPreviewAlert.keyWord", { count: impactResult.affected_keys_count }),
                  teamsCount: impactResult.affected_teams_count,
                  teamsWord: t("policies.impactPreviewAlert.teamWord", { count: impactResult.affected_teams_count }),
                }}
                components={[<strong key="keys" />, <strong key="teams" />]}
              />
            </Text>
            {impactResult.sample_keys.length > 0 && (
              <div className="mt-1">
                <Text type="secondary" style={{ fontSize: 12 }}>
                  {t("policies.impactPreviewAlert.keysLabel")}{" "}
                </Text>
                {impactResult.sample_keys.slice(0, 5).map((k: string) => (
                  <Tag key={k} style={{ fontSize: 11 }}>
                    {k}
                  </Tag>
                ))}
                {impactResult.affected_keys_count > 5 && (
                  <Text type="secondary" style={{ fontSize: 11 }}>
                    {t("policies.impactPreviewAlert.andMore", { count: impactResult.affected_keys_count - 5 })}
                  </Text>
                )}
              </div>
            )}
            {impactResult.sample_teams.length > 0 && (
              <div className="mt-1">
                <Text type="secondary" style={{ fontSize: 12 }}>
                  {t("policies.impactPreviewAlert.teamsLabel")}{" "}
                </Text>
                {impactResult.sample_teams.slice(0, 5).map((teamName: string) => (
                  <Tag key={teamName} style={{ fontSize: 11 }}>
                    {teamName}
                  </Tag>
                ))}
                {impactResult.affected_teams_count > 5 && (
                  <Text type="secondary" style={{ fontSize: 11 }}>
                    {t("policies.impactPreviewAlert.andMore", { count: impactResult.affected_teams_count - 5 })}
                  </Text>
                )}
              </div>
            )}
          </div>
        )
      }
    />
  );
};

export default ImpactPreviewAlert;
