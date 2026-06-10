import React, { useState, useEffect, useCallback } from "react";
import { Card, Badge, Button } from "@tremor/react";
import { ArrowLeftIcon, PencilIcon } from "@heroicons/react/outline";
import { Descriptions, Tag, Spin, Divider, Typography, Alert } from "antd";
import { useTranslation } from "react-i18next";
import { Policy } from "./types";
import { PipelineInfoDisplay } from "./pipeline_flow_builder";
import { getResolvedGuardrails } from "../networking";

const { Title, Text } = Typography;

interface PolicyInfoViewProps {
  policyId: string;
  onClose: () => void;
  onEdit: (policy: Policy) => void;
  accessToken: string | null;
  isAdmin: boolean;
  getPolicy: (accessToken: string, policyId: string) => Promise<any>;
}

const PolicyInfoView: React.FC<PolicyInfoViewProps> = ({
  policyId,
  onClose,
  onEdit,
  accessToken,
  isAdmin,
  getPolicy,
}) => {
  const { t } = useTranslation();
  const [policy, setPolicy] = useState<Policy | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [resolvedGuardrails, setResolvedGuardrails] = useState<string[]>([]);
  const [isLoadingResolved, setIsLoadingResolved] = useState(false);

  const fetchPolicy = useCallback(async () => {
    if (!accessToken || !policyId) return;

    setIsLoading(true);
    try {
      const data = await getPolicy(accessToken, policyId);
      setPolicy(data);

      // Also fetch resolved guardrails
      setIsLoadingResolved(true);
      try {
        const resolvedData = await getResolvedGuardrails(accessToken, policyId);
        setResolvedGuardrails(resolvedData.resolved_guardrails || []);
      } catch (error) {
        console.error("Error fetching resolved guardrails:", error);
      } finally {
        setIsLoadingResolved(false);
      }
    } catch (error) {
      console.error("Error fetching policy:", error);
    } finally {
      setIsLoading(false);
    }
  }, [policyId, accessToken, getPolicy]);

  useEffect(() => {
    fetchPolicy();
  }, [fetchPolicy]);

  if (isLoading) {
    return (
      <div className="flex justify-center items-center p-12">
        <Spin size="large" />
      </div>
    );
  }

  if (!policy) {
    return (
      <Card>
        <Text type="danger">{t("policies.policyInfo.notFound")}</Text>
        <br />
        <Button onClick={onClose} className="mt-4">
          {t("common.back")}
        </Button>
      </Card>
    );
  }

  return (
    <Card>
      <div className="space-y-6">
        <div className="flex justify-between items-center">
          <Button variant="secondary" icon={ArrowLeftIcon} onClick={onClose}>
            {t("policies.policyInfo.backToPolicies")}
          </Button>
          {isAdmin && (
            <Button icon={PencilIcon} onClick={() => onEdit(policy)}>
              {t("policies.policyInfo.editPolicy")}
            </Button>
          )}
        </div>

        <Title level={4}>{policy.policy_name}</Title>

        <Descriptions bordered column={1}>
          <Descriptions.Item label={t("policies.policyInfo.policyId")}>
            <code className="text-xs bg-gray-100 px-2 py-1 rounded">{policy.policy_id}</code>
          </Descriptions.Item>
          <Descriptions.Item label={t("common.description")}>
            {policy.description || <Text type="secondary">{t("policies.policyInfo.noDescription")}</Text>}
          </Descriptions.Item>
          <Descriptions.Item label={t("policies.policyInfo.inheritsFrom")}>
            {policy.inherit ? (
              <Badge color="blue" size="sm">
                {policy.inherit}
              </Badge>
            ) : (
              <Text type="secondary">{t("common.none")}</Text>
            )}
          </Descriptions.Item>
          <Descriptions.Item label={t("common.createdAt")}>
            {policy.created_at ? new Date(policy.created_at).toLocaleString() : "-"}
          </Descriptions.Item>
          <Descriptions.Item label={t("common.updatedAt")}>
            {policy.updated_at ? new Date(policy.updated_at).toLocaleString() : "-"}
          </Descriptions.Item>
        </Descriptions>

        {policy.pipeline && (
          <>
            <Divider orientation="left">
              <Text strong>{t("policies.policyInfo.pipelineFlow")}</Text>
            </Divider>
            <Alert
              message={t("policies.policyInfo.pipelineInfo", {
                count: policy.pipeline.steps.length,
                mode: policy.pipeline.mode,
              })}
              type="info"
              showIcon
              style={{ marginBottom: 16 }}
            />
            <PipelineInfoDisplay pipeline={policy.pipeline} />
          </>
        )}

        <Divider orientation="left">
          <Text strong>{t("policies.policyInfo.guardrailsConfiguration")}</Text>
        </Divider>

        {resolvedGuardrails.length > 0 && (
          <Alert
            message={t("policies.policyInfo.resolvedGuardrails")}
            description={
              <div>
                <Text type="secondary" style={{ display: "block", marginBottom: 8 }}>
                  {t("policies.policyInfo.resolvedGuardrailsDesc")}
                </Text>
                <div className="flex flex-wrap gap-1">
                  {resolvedGuardrails.map((g) => (
                    <Tag key={g} color="blue">
                      {g}
                    </Tag>
                  ))}
                </div>
              </div>
            }
            type="info"
            showIcon
            style={{ marginBottom: 16 }}
          />
        )}

        <Descriptions bordered column={1}>
          <Descriptions.Item label={t("policies.policyInfo.guardrailsToAdd")}>
            <div className="flex flex-wrap gap-1">
              {policy.guardrails_add && policy.guardrails_add.length > 0 ? (
                policy.guardrails_add.map((g) => (
                  <Tag key={g} color="green">
                    {g}
                  </Tag>
                ))
              ) : (
                <Text type="secondary">{t("common.none")}</Text>
              )}
            </div>
          </Descriptions.Item>
          <Descriptions.Item label={t("policies.policyInfo.guardrailsToRemove")}>
            <div className="flex flex-wrap gap-1">
              {policy.guardrails_remove && policy.guardrails_remove.length > 0 ? (
                policy.guardrails_remove.map((g) => (
                  <Tag key={g} color="red">
                    {g}
                  </Tag>
                ))
              ) : (
                <Text type="secondary">{t("common.none")}</Text>
              )}
            </div>
          </Descriptions.Item>
        </Descriptions>

        <Divider orientation="left">
          <Text strong>{t("policies.policyInfo.conditions")}</Text>
        </Divider>

        <Descriptions bordered column={1}>
          <Descriptions.Item label={t("policies.policyInfo.modelCondition")}>
            {policy.condition?.model ? (
              <Tag color="purple">
                {typeof policy.condition.model === "string"
                  ? policy.condition.model
                  : JSON.stringify(policy.condition.model)}
              </Tag>
            ) : (
              <Text type="secondary">{t("policies.policyInfo.noModelCondition")}</Text>
            )}
          </Descriptions.Item>
        </Descriptions>
      </div>
    </Card>
  );
};

export default PolicyInfoView;
