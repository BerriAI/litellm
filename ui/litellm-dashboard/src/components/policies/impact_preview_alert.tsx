import React from "react";
import { Alert, Tag, Typography } from "antd";

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
  return (
    <Alert
      type={impactResult.affected_keys_count === -1 ? "warning" : "info"}
      showIcon
      className="mb-4"
      message="Impact Preview"
      description={
        impactResult.affected_keys_count === -1 ? (
          <Text>Global scope — this will affect <strong>all keys and teams</strong>.</Text>
        ) : (
          <div>
            <Text>
              This attachment would affect <strong>{impactResult.affected_keys_count} key{impactResult.affected_keys_count !== 1 ? "s" : ""}</strong> and <strong>{impactResult.affected_teams_count} team{impactResult.affected_teams_count !== 1 ? "s" : ""}</strong>.
            </Text>
            {impactResult.sample_keys.length > 0 && (
              <div className="mt-1">
                <Text type="secondary" style={{ fontSize: 12 }}>Keys: </Text>
                {impactResult.sample_keys.slice(0, 5).map((k: string) => (
                  <Tag key={k} style={{ fontSize: 11 }}>{k}</Tag>
                ))}
                {impactResult.affected_keys_count > 5 && (
                  <Text type="secondary" style={{ fontSize: 11 }}>and {impactResult.affected_keys_count - 5} more...</Text>
                )}
              </div>
            )}
            {impactResult.sample_teams.length > 0 && (
              <div className="mt-1">
                <Text type="secondary" style={{ fontSize: 12 }}>Teams: </Text>
                {impactResult.sample_teams.slice(0, 5).map((t: string) => (
                  <Tag key={t} style={{ fontSize: 11 }}>{t}</Tag>
                ))}
                {impactResult.affected_teams_count > 5 && (
                  <Text type="secondary" style={{ fontSize: 11 }}>and {impactResult.affected_teams_count - 5} more...</Text>
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
