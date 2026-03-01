"use client";

import React, { useEffect, useState } from "react";
import {
  Button,
  Collapse,
  Form,
  Input,
  InputNumber,
  Modal,
  Radio,
  Space,
  Typography,
} from "antd";
import { CloseOutlined, PlusOutlined } from "@ant-design/icons";
import { routingGroupCreateCall, routingGroupUpdateCall } from "@/components/networking";

interface Deployment {
  model_name: string;
  litellm_provider: string;
  weight?: number;
  priority?: number;
}

interface RoutingGroupBuilderProps {
  accessToken: string | null;
  visible: boolean;
  editTarget: Record<string, unknown> | null;
  onClose: () => void;
  onSuccess: () => void;
}

const ROUTING_STRATEGIES = [
  { value: "priority-failover", label: "Priority Failover: ordered, first success wins" },
  { value: "weighted", label: "Weighted: split traffic by percentage" },
  { value: "cost-based-routing", label: "Cost-Based: route to cheapest" },
  { value: "latency-based-routing", label: "Latency-Based: route to fastest" },
  { value: "least-busy", label: "Least Busy: fewest in-flight requests" },
  { value: "usage-based-routing-v2", label: "Usage-Based: TPM/RPM aware" },
  { value: "simple-shuffle", label: "Round Robin: even distribution" },
];

export default function RoutingGroupBuilder({
  accessToken,
  visible,
  editTarget,
  onClose,
  onSuccess,
}: RoutingGroupBuilderProps) {
  const [form] = Form.useForm();
  const [deployments, setDeployments] = useState<Deployment[]>([]);
  const [newModelName, setNewModelName] = useState("");
  const [newProvider, setNewProvider] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [strategy, setStrategy] = useState<string>("priority-failover");

  const isEditing = editTarget !== null;

  useEffect(() => {
    if (editTarget) {
      form.setFieldsValue({
        routing_group_name: editTarget.routing_group_name,
        description: editTarget.description,
        routing_strategy: editTarget.routing_strategy ?? "priority-failover",
        max_retries: editTarget.max_retries,
        cooldown_time: editTarget.cooldown_time,
      });
      setStrategy((editTarget.routing_strategy as string) ?? "priority-failover");
      if (Array.isArray(editTarget.deployments)) {
        setDeployments(editTarget.deployments as Deployment[]);
      }
    } else {
      form.resetFields();
      setDeployments([]);
      setStrategy("priority-failover");
    }
  }, [editTarget, form]);

  const addDeployment = () => {
    const trimmedModel = newModelName.trim();
    if (!trimmedModel) return;
    const newDeployment: Deployment = {
      model_name: trimmedModel,
      litellm_provider: newProvider.trim(),
      weight: strategy === "weighted" ? 1 : undefined,
      priority: strategy === "priority-failover" ? deployments.length + 1 : undefined,
    };
    setDeployments((prev) => [...prev, newDeployment]);
    setNewModelName("");
    setNewProvider("");
  };

  const removeDeployment = (index: number) => {
    setDeployments((prev) => {
      const updated = prev.filter((_, i) => i !== index);
      if (strategy === "priority-failover") {
        return updated.map((d, i) => ({ ...d, priority: i + 1 }));
      }
      return updated;
    });
  };

  const updateDeploymentWeight = (index: number, weight: number | null) => {
    setDeployments((prev) =>
      prev.map((d, i) => (i === index ? { ...d, weight: weight ?? 1 } : d))
    );
  };

  const handleStrategyChange = (value: string) => {
    setStrategy(value);
    setDeployments((prev) =>
      prev.map((d, i) => ({
        ...d,
        weight: value === "weighted" ? (d.weight ?? 1) : undefined,
        priority: value === "priority-failover" ? i + 1 : undefined,
      }))
    );
  };

  const handleSubmit = async () => {
    if (!accessToken) return;
    try {
      const values = await form.validateFields();
      setSubmitting(true);

      const payload: Record<string, unknown> = {
        routing_group_name: values.routing_group_name,
        description: values.description,
        routing_strategy: values.routing_strategy,
        deployments,
        max_retries: values.max_retries,
        cooldown_time: values.cooldown_time,
      };

      if (isEditing && editTarget?.routing_group_id) {
        await routingGroupUpdateCall(
          accessToken,
          editTarget.routing_group_id as string,
          payload
        );
      } else {
        await routingGroupCreateCall(accessToken, payload);
      }

      onSuccess();
    } catch (err) {
      console.error("Failed to save routing group:", err);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Modal
      title={isEditing ? "Edit Routing Group" : "Create Routing Group"}
      open={visible}
      onCancel={onClose}
      width={700}
      footer={[
        <Button key="cancel" onClick={onClose}>
          Cancel
        </Button>,
        <Button
          key="submit"
          type="primary"
          loading={submitting}
          onClick={handleSubmit}
        >
          {isEditing ? "Save Changes" : "Create"}
        </Button>,
      ]}
    >
      <Form
        form={form}
        layout="vertical"
        initialValues={{ routing_strategy: "priority-failover" }}
      >
        <Form.Item
          name="routing_group_name"
          label="Name"
          rules={[{ required: true, message: "Please enter a name" }]}
        >
          <Input placeholder="e.g. production-gpt4-pool" />
        </Form.Item>

        <Form.Item name="description" label="Description">
          <Input.TextArea rows={2} placeholder="Optional description" />
        </Form.Item>

        <Form.Item
          name="routing_strategy"
          label="Routing Strategy"
          rules={[{ required: true, message: "Please select a strategy" }]}
        >
          <Radio.Group
            onChange={(e) => handleStrategyChange(e.target.value)}
          >
            <Space direction="vertical">
              {ROUTING_STRATEGIES.map((s) => (
                <Radio key={s.value} value={s.value}>
                  {s.label}
                </Radio>
              ))}
            </Space>
          </Radio.Group>
        </Form.Item>

        <div style={{ marginBottom: 16 }}>
          <Typography.Text strong>Deployments</Typography.Text>
          <div style={{ marginTop: 8, marginBottom: 8 }}>
            <Space>
              <Input
                placeholder="Model name (e.g. gpt-4)"
                value={newModelName}
                onChange={(e) => setNewModelName(e.target.value)}
                style={{ width: 200 }}
                onPressEnter={addDeployment}
              />
              <Input
                placeholder="Provider (e.g. openai)"
                value={newProvider}
                onChange={(e) => setNewProvider(e.target.value)}
                style={{ width: 160 }}
                onPressEnter={addDeployment}
              />
              <Button icon={<PlusOutlined />} onClick={addDeployment}>
                Add Deployment
              </Button>
            </Space>
          </div>

          {deployments.length === 0 && (
            <Typography.Text type="secondary">
              No deployments added yet.
            </Typography.Text>
          )}

          {deployments.map((dep, idx) => (
            <div
              key={idx}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                marginBottom: 6,
                padding: "6px 10px",
                background: "#fafafa",
                border: "1px solid #f0f0f0",
                borderRadius: 4,
              }}
            >
              {strategy === "priority-failover" && (
                <Typography.Text type="secondary" style={{ minWidth: 24 }}>
                  #{dep.priority}
                </Typography.Text>
              )}
              <Typography.Text style={{ flex: 1 }}>
                {dep.model_name}
                {dep.litellm_provider ? ` (${dep.litellm_provider})` : ""}
              </Typography.Text>
              {strategy === "weighted" && (
                <InputNumber
                  size="small"
                  min={1}
                  max={100}
                  value={dep.weight ?? 1}
                  onChange={(val) => updateDeploymentWeight(idx, val)}
                  addonAfter="%"
                  style={{ width: 100 }}
                />
              )}
              <Button
                size="small"
                type="text"
                danger
                icon={<CloseOutlined />}
                onClick={() => removeDeployment(idx)}
              />
            </div>
          ))}
        </div>

        <Collapse ghost>
          <Collapse.Panel header="Advanced Settings" key="advanced">
            <Form.Item name="max_retries" label="Max Retries">
              <InputNumber min={0} max={100} style={{ width: 160 }} />
            </Form.Item>
            <Form.Item name="cooldown_time" label="Cooldown Time (seconds)">
              <InputNumber min={0} style={{ width: 160 }} />
            </Form.Item>
          </Collapse.Panel>
        </Collapse>

        <div
          style={{
            marginTop: 12,
            padding: "8px 12px",
            background: "#f6f8fa",
            borderRadius: 4,
          }}
        >
          <Typography.Text type="secondary">
            Access is managed via Teams.{" "}
            <a href="/teams" target="_blank" rel="noreferrer">
              Manage Teams
            </a>
          </Typography.Text>
        </div>
      </Form>
    </Modal>
  );
}
