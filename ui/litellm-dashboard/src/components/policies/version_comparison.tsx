import React, { useState, useEffect } from "react";
import { Card } from "@tremor/react";
import { Typography, Spin, Tag, Divider, Alert } from "antd";
import { PolicyVersionComparison } from "./types";
import VersionStatusBadge from "./version_status_badge";
import { comparePolicyVersions } from "../networking";
import NotificationsManager from "../molecules/notifications_manager";

const { Title, Text } = Typography;

interface VersionComparisonProps {
  policyId1: string;
  policyId2: string;
  accessToken: string | null;
}

const VersionComparison: React.FC<VersionComparisonProps> = ({
  policyId1,
  policyId2,
  accessToken,
}) => {
  const [comparison, setComparison] = useState<PolicyVersionComparison | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const loadComparison = async () => {
      if (!accessToken || !policyId1 || !policyId2) return;

      setIsLoading(true);
      try {
        const data = await comparePolicyVersions(accessToken, policyId1, policyId2);
        setComparison(data);
      } catch (error) {
        console.error("Failed to compare versions:", error);
        NotificationsManager.fromBackend("Failed to compare policy versions");
      } finally {
        setIsLoading(false);
      }
    };

    loadComparison();
  }, [policyId1, policyId2, accessToken]);

  if (isLoading) {
    return (
      <div className="flex justify-center items-center p-12">
        <Spin size="large" />
      </div>
    );
  }

  if (!comparison) {
    return (
      <Alert
        message="Failed to load comparison"
        description="Could not compare the selected versions."
        type="error"
        showIcon
      />
    );
  }

  const { policy_1, policy_2, differences } = comparison;

  const renderFieldComparison = (
    label: string,
    oldValue: any,
    newValue: any,
    changed: boolean
  ) => {
    if (!changed) {
      return (
        <div className="mb-4">
          <Text strong style={{ fontSize: 14, display: "block", marginBottom: 8 }}>
            {label}
          </Text>
          <div className="p-3 rounded bg-gray-50 border border-gray-200">
            <Text style={{ fontSize: 13 }}>{oldValue || "(Not set)"}</Text>
          </div>
        </div>
      );
    }

    return (
      <div className="mb-4">
        <Text strong style={{ fontSize: 14, display: "block", marginBottom: 8 }}>
          {label}
        </Text>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="mb-2">
              <Tag color="red" style={{ fontSize: 11 }}>
                v{policy_2.version_number} (Old)
              </Tag>
            </div>
            <div className="p-3 rounded bg-red-50 border border-red-200">
              <Text style={{ fontSize: 13 }}>{oldValue || "(Not set)"}</Text>
            </div>
          </div>
          <div>
            <div className="mb-2">
              <Tag color="green" style={{ fontSize: 11 }}>
                v{policy_1.version_number} (New)
              </Tag>
            </div>
            <div className="p-3 rounded bg-green-50 border border-green-200">
              <Text style={{ fontSize: 13 }}>{newValue || "(Not set)"}</Text>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderArrayFieldComparison = (
    label: string,
    added: string[],
    removed: string[],
    unchanged: string[]
  ) => {
    const hasChanges = added.length > 0 || removed.length > 0;

    if (!hasChanges && unchanged.length === 0) {
      return (
        <div className="mb-4">
          <Text strong style={{ fontSize: 14, display: "block", marginBottom: 8 }}>
            {label}
          </Text>
          <div className="p-3 rounded bg-gray-50 border border-gray-200">
            <Text type="secondary" style={{ fontSize: 13 }}>
              (None)
            </Text>
          </div>
        </div>
      );
    }

    return (
      <div className="mb-4">
        <Text strong style={{ fontSize: 14, display: "block", marginBottom: 8 }}>
          {label}
        </Text>
        <div className="p-3 rounded bg-gray-50 border border-gray-200">
          <div className="space-y-2">
            {added.length > 0 && (
              <div>
                <Text type="secondary" style={{ fontSize: 12, display: "block", marginBottom: 4 }}>
                  Added:
                </Text>
                <div className="flex flex-wrap gap-1">
                  {added.map((item) => (
                    <Tag key={item} color="green" style={{ fontSize: 12 }}>
                      + {item}
                    </Tag>
                  ))}
                </div>
              </div>
            )}
            {removed.length > 0 && (
              <div>
                <Text type="secondary" style={{ fontSize: 12, display: "block", marginBottom: 4 }}>
                  Removed:
                </Text>
                <div className="flex flex-wrap gap-1">
                  {removed.map((item) => (
                    <Tag key={item} color="red" style={{ fontSize: 12 }}>
                      - {item}
                    </Tag>
                  ))}
                </div>
              </div>
            )}
            {unchanged.length > 0 && (
              <div>
                <Text type="secondary" style={{ fontSize: 12, display: "block", marginBottom: 4 }}>
                  Unchanged:
                </Text>
                <div className="flex flex-wrap gap-1">
                  {unchanged.map((item) => (
                    <Tag key={item} color="default" style={{ fontSize: 12 }}>
                      {item}
                    </Tag>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  return (
    <Card>
      <div className="space-y-6">
        {/* Header */}
        <div>
          <Title level={4} style={{ marginBottom: 16 }}>
            Version Comparison
          </Title>
          <div className="grid grid-cols-2 gap-4">
            <div className="p-4 rounded-lg bg-blue-50 border border-blue-200">
              <div className="flex items-center gap-2 mb-2">
                <Text strong style={{ fontSize: 14 }}>
                  Version {policy_1.version_number}
                </Text>
                <VersionStatusBadge status={policy_1.version_status as any} size="xs" />
              </div>
              <Text type="secondary" style={{ fontSize: 12 }}>
                {policy_1.policy_name}
              </Text>
            </div>
            <div className="p-4 rounded-lg bg-gray-50 border border-gray-200">
              <div className="flex items-center gap-2 mb-2">
                <Text strong style={{ fontSize: 14 }}>
                  Version {policy_2.version_number}
                </Text>
                <VersionStatusBadge status={policy_2.version_status as any} size="xs" />
              </div>
              <Text type="secondary" style={{ fontSize: 12 }}>
                {policy_2.policy_name}
              </Text>
            </div>
          </div>
        </div>

        <Divider />

        {/* Differences */}
        <div>
          <Text strong style={{ fontSize: 15, display: "block", marginBottom: 16 }}>
            Changes
          </Text>

          {/* Description */}
          {differences.description && (
            <>
              {renderFieldComparison(
                "Description",
                differences.description.old,
                differences.description.new,
                differences.description.changed
              )}
            </>
          )}

          {/* Inherit */}
          {differences.inherit && (
            <>
              {renderFieldComparison(
                "Inherits From",
                differences.inherit.old,
                differences.inherit.new,
                differences.inherit.changed
              )}
            </>
          )}

          {/* Guardrails Add */}
          {differences.guardrails_add && (
            <>
              {renderArrayFieldComparison(
                "Guardrails to Add",
                differences.guardrails_add.added,
                differences.guardrails_add.removed,
                differences.guardrails_add.unchanged
              )}
            </>
          )}

          {/* Guardrails Remove */}
          {differences.guardrails_remove && (
            <>
              {renderArrayFieldComparison(
                "Guardrails to Remove",
                differences.guardrails_remove.added,
                differences.guardrails_remove.removed,
                differences.guardrails_remove.unchanged
              )}
            </>
          )}

          {/* Condition */}
          {differences.condition && (
            <>
              {renderFieldComparison(
                "Model Condition",
                differences.condition.old?.model || "(None)",
                differences.condition.new?.model || "(None)",
                differences.condition.changed
              )}
            </>
          )}

          {/* Pipeline */}
          {differences.pipeline && (
            <>
              {renderFieldComparison(
                "Pipeline",
                differences.pipeline.old ? JSON.stringify(differences.pipeline.old, null, 2) : "(None)",
                differences.pipeline.new ? JSON.stringify(differences.pipeline.new, null, 2) : "(None)",
                differences.pipeline.changed
              )}
            </>
          )}

          {/* No changes message */}
          {Object.keys(differences).length === 0 && (
            <Alert
              message="No differences found"
              description="These versions are identical."
              type="info"
              showIcon
            />
          )}
        </div>
      </div>
    </Card>
  );
};

export default VersionComparison;
