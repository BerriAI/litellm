import React, { useState, useEffect } from "react";
import {
  Card,
  Descriptions,
  Tag,
  Button,
  Space,
  Typography,
  Spin,
  Divider,
} from "antd";
import { ArrowLeftOutlined, EditOutlined } from "@ant-design/icons";
import { Policy } from "./types";

const { Title, Text } = Typography;

interface PolicyInfoViewProps {
  policyId: string;
  onClose: () => void;
  onEdit: (policy: Policy) => void;
  accessToken: string | null;
  isAdmin: boolean;
  fetchPolicy: (policyId: string) => Promise<Policy | null>;
}

const PolicyInfoView: React.FC<PolicyInfoViewProps> = ({
  policyId,
  onClose,
  onEdit,
  accessToken,
  isAdmin,
  fetchPolicy,
}) => {
  const [policy, setPolicy] = useState<Policy | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const loadPolicy = async () => {
      if (!accessToken || !policyId) return;

      setIsLoading(true);
      try {
        const data = await fetchPolicy(policyId);
        setPolicy(data);
      } catch (error) {
        console.error("Error fetching policy:", error);
      } finally {
        setIsLoading(false);
      }
    };

    loadPolicy();
  }, [policyId, accessToken, fetchPolicy]);

  if (isLoading) {
    return (
      <div style={{ textAlign: "center", padding: "50px" }}>
        <Spin size="large" />
      </div>
    );
  }

  if (!policy) {
    return (
      <Card>
        <Text type="danger">Policy not found</Text>
        <br />
        <Button onClick={onClose} style={{ marginTop: 16 }}>
          Go Back
        </Button>
      </Card>
    );
  }

  return (
    <Card>
      <Space direction="vertical" style={{ width: "100%" }} size="large">
        <Space style={{ justifyContent: "space-between", width: "100%" }}>
          <Button icon={<ArrowLeftOutlined />} onClick={onClose}>
            Back to Policies
          </Button>
          {isAdmin && (
            <Button
              type="primary"
              icon={<EditOutlined />}
              onClick={() => onEdit(policy)}
            >
              Edit Policy
            </Button>
          )}
        </Space>

        <Title level={4}>{policy.policy_name}</Title>

        <Descriptions bordered column={1}>
          <Descriptions.Item label="Policy ID">
            <Text code>{policy.policy_id}</Text>
          </Descriptions.Item>
          <Descriptions.Item label="Description">
            {policy.description || <Text type="secondary">No description</Text>}
          </Descriptions.Item>
          <Descriptions.Item label="Inherits From">
            {policy.inherit ? (
              <Tag color="blue">{policy.inherit}</Tag>
            ) : (
              <Text type="secondary">None</Text>
            )}
          </Descriptions.Item>
          <Descriptions.Item label="Created At">
            {policy.created_at
              ? new Date(policy.created_at).toLocaleString()
              : "-"}
          </Descriptions.Item>
          <Descriptions.Item label="Updated At">
            {policy.updated_at
              ? new Date(policy.updated_at).toLocaleString()
              : "-"}
          </Descriptions.Item>
        </Descriptions>

        <Divider orientation="left">Guardrails Configuration</Divider>

        <Descriptions bordered column={1}>
          <Descriptions.Item label="Guardrails to Add">
            <Space wrap>
              {policy.guardrails_add && policy.guardrails_add.length > 0 ? (
                policy.guardrails_add.map((g) => (
                  <Tag key={g} color="green">
                    {g}
                  </Tag>
                ))
              ) : (
                <Text type="secondary">None</Text>
              )}
            </Space>
          </Descriptions.Item>
          <Descriptions.Item label="Guardrails to Remove">
            <Space wrap>
              {policy.guardrails_remove && policy.guardrails_remove.length > 0 ? (
                policy.guardrails_remove.map((g) => (
                  <Tag key={g} color="red">
                    {g}
                  </Tag>
                ))
              ) : (
                <Text type="secondary">None</Text>
              )}
            </Space>
          </Descriptions.Item>
        </Descriptions>

        <Divider orientation="left">Conditions</Divider>

        <Descriptions bordered column={1}>
          <Descriptions.Item label="Model Condition">
            {policy.condition?.model ? (
              <Tag color="purple">{policy.condition.model}</Tag>
            ) : (
              <Text type="secondary">No model condition (applies to all models)</Text>
            )}
          </Descriptions.Item>
        </Descriptions>
      </Space>
    </Card>
  );
};

export default PolicyInfoView;
