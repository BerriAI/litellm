"use client";
import React, { useEffect, useState } from "react";
import {
  Button,
  Table,
  Tag,
  Typography,
  Space,
  Drawer,
  Form,
  Input,
  InputNumber,
  Select,
  Popconfirm,
  message,
} from "antd";
import { PlusOutlined, DeleteOutlined } from "@ant-design/icons";
import { createLiteLLMEval, listLiteLLMEvals, deleteLiteLLMEval, modelAvailableCall } from "../networking";

const { Title, Text } = Typography;

interface EvalCriterion {
  name: string;
  weight: number;
  description: string;
  threshold?: number;
}

interface LiteLLMEval {
  eval_id: string;
  eval_name: string;
  version: number;
  criteria: EvalCriterion[];
  judge_model: string;
  description?: string;
  overall_threshold?: number;
  max_iterations: number;
  created_at: string;
}

interface Props {
  accessToken: string | null;
  userRole?: string | null;
  availableModels?: string[];
}

export default function EvalsCatalog({ accessToken, userRole, availableModels: availableModelsProp = [] }: Props) {
  const [evals, setEvals] = useState<LiteLLMEval[]>([]);
  const [loading, setLoading] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [saving, setSaving] = useState(false);
  const [availableModels, setAvailableModels] = useState<string[]>(availableModelsProp);
  const [form] = Form.useForm();

  const fetchEvals = async () => {
    if (!accessToken) return;
    setLoading(true);
    try {
      const data = await listLiteLLMEvals(accessToken);
      setEvals(data);
    } catch (e: any) {
      message.error(`Failed to load evals: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  const fetchModels = async () => {
    if (!accessToken || availableModelsProp.length > 0) return;
    try {
      const data = await modelAvailableCall(accessToken, null, null);
      const names: string[] = data?.data?.map((m: any) => m.id) ?? [];
      setAvailableModels(names);
    } catch {
      // best-effort; leave empty
    }
  };

  useEffect(() => {
    fetchEvals();
    fetchModels();
  }, [accessToken]);

  const handleCreate = async (values: any) => {
    if (!accessToken) return;
    setSaving(true);
    try {
      await createLiteLLMEval(accessToken, {
        eval_name: values.eval_name,
        criteria: values.criteria || [],
        judge_model: values.judge_model,
        description: values.description,
        overall_threshold: values.overall_threshold ?? 80,
        max_iterations: values.max_iterations ?? 1,
      });
      message.success("Eval created");
      setDrawerOpen(false);
      form.resetFields();
      fetchEvals();
    } catch (e: any) {
      message.error(`Failed to create eval: ${e.message}`);
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (evalId: string) => {
    if (!accessToken) return;
    try {
      await deleteLiteLLMEval(accessToken, evalId);
      message.success("Eval deleted");
      fetchEvals();
    } catch (e: any) {
      message.error(`Failed to delete eval: ${e.message}`);
    }
  };

  const columns = [
    {
      title: "Name",
      dataIndex: "eval_name",
      key: "eval_name",
      render: (v: string) => <Text strong>{v}</Text>,
    },
    {
      title: "Judge Model",
      dataIndex: "judge_model",
      key: "judge_model",
    },
    {
      title: "Minimum Score to Pass",
      dataIndex: "overall_threshold",
      key: "overall_threshold",
      render: (v?: number) =>
        v != null ? <Tag color="blue">≥ {v} / 100</Tag> : <Text type="secondary">—</Text>,
    },
    {
      title: "Criteria",
      dataIndex: "criteria",
      key: "criteria",
      render: (v: EvalCriterion[]) => <Tag>{v?.length ?? 0} criteria</Tag>,
    },
    {
      title: "Version",
      dataIndex: "version",
      key: "version",
      render: (v: number) => <Text type="secondary">v{v}</Text>,
    },
    {
      title: "Actions",
      key: "actions",
      render: (_: any, record: LiteLLMEval) => (
        <Popconfirm
          title="Delete this eval?"
          onConfirm={() => handleDelete(record.eval_id)}
          okText="Delete"
          okButtonProps={{ danger: true }}
        >
          <Button danger icon={<DeleteOutlined />} size="small" />
        </Popconfirm>
      ),
    },
  ];

  return (
    <div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 16,
        }}
      >
        <Space>
          <Title level={4} style={{ margin: 0 }}>
            Evals
          </Title>
          <Tag color="blue">Beta</Tag>
        </Space>
        <Button type="primary" icon={<PlusOutlined />} onClick={() => setDrawerOpen(true)}>
          Create Eval
        </Button>
      </div>

      <Table
        dataSource={evals}
        columns={columns}
        rowKey="eval_id"
        loading={loading}
        locale={{ emptyText: "No evals yet. Create one to get started." }}
      />

      <Drawer
        title={
          <Space>
            <span>Create Eval</span>
            <Tag color="blue">Beta</Tag>
          </Space>
        }
        open={drawerOpen}
        onClose={() => {
          setDrawerOpen(false);
          form.resetFields();
        }}
        width={560}
        footer={
          <div style={{ textAlign: "right" }}>
            <Button style={{ marginRight: 8 }} onClick={() => setDrawerOpen(false)}>
              Cancel
            </Button>
            <Button type="primary" loading={saving} onClick={() => form.submit()}>
              Create Eval
            </Button>
          </div>
        }
      >
        <Form form={form} layout="vertical" onFinish={handleCreate}>
          <Form.Item name="eval_name" label="Eval Name" rules={[{ required: true }]}>
            <Input placeholder="e.g. Insurance Claims QA" />
          </Form.Item>

          <Form.Item name="description" label="Description">
            <Input.TextArea rows={2} placeholder="What does this eval measure?" />
          </Form.Item>

          <Form.Item name="judge_model" label="Judge Model" rules={[{ required: true }]}>
            <Select
              showSearch
              placeholder="Select a model"
              options={availableModels.map((m) => ({ label: m, value: m }))}
            />
          </Form.Item>

          <Form.Item
            name="overall_threshold"
            label="Minimum Score to Pass"
            initialValue={80}
          >
            <InputNumber min={0} max={100} addonAfter="/ 100" style={{ width: "100%" }} />
          </Form.Item>

          <Form.Item label="Evaluation Criteria">
            <Form.List name="criteria">
              {(fields, { add, remove }) => (
                <>
                  {fields.map(({ key, name, ...restField }) => (
                    <div
                      key={key}
                      style={{
                        border: "1px solid #f0f0f0",
                        borderRadius: 6,
                        padding: "12px 12px 0",
                        marginBottom: 8,
                      }}
                    >
                      <div style={{ display: "flex", gap: 8 }}>
                        <Form.Item
                          {...restField}
                          name={[name, "name"]}
                          rules={[{ required: true, message: "Enter criterion name" }]}
                          style={{ flex: 2, marginBottom: 8 }}
                        >
                          <Input placeholder="Criterion name (e.g. Policy accuracy)" />
                        </Form.Item>
                        <Form.Item
                          {...restField}
                          name={[name, "weight"]}
                          rules={[{ required: true, message: "Enter weight" }]}
                          style={{ flex: 1, marginBottom: 8 }}
                        >
                          <InputNumber
                            min={0}
                            max={100}
                            addonAfter="%"
                            style={{ width: "100%" }}
                            placeholder="Weight"
                          />
                        </Form.Item>
                        <Button danger onClick={() => remove(name)} style={{ marginTop: 0 }}>
                          ×
                        </Button>
                      </div>
                      <Form.Item
                        {...restField}
                        name={[name, "description"]}
                        rules={[{ required: true, message: "Describe what to check" }]}
                        style={{ marginBottom: 8 }}
                      >
                        <Input placeholder="What should the judge check for this criterion?" />
                      </Form.Item>
                    </div>
                  ))}
                  <Button
                    type="dashed"
                    onClick={() => add({ weight: 0 })}
                    block
                    icon={<PlusOutlined />}
                  >
                    Add Criterion
                  </Button>
                </>
              )}
            </Form.List>
          </Form.Item>

          <Form.Item
            name="max_iterations"
            label="Retry on fail (attempts)"
            initialValue={1}
          >
            <InputNumber min={1} max={5} style={{ width: "100%" }} />
          </Form.Item>
        </Form>
      </Drawer>
    </div>
  );
}
