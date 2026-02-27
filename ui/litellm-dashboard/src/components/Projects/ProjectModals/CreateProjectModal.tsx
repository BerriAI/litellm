import { useEffect, useState } from "react";
import {
  Alert,
  Modal,
  Form,
  Input,
  Select,
  Switch,
  InputNumber,
  Collapse,
  Button,
  Col,
  Flex,
  Row,
  Space,
  Divider,
  Typography,
  message,
} from "antd";
import { FolderAddOutlined, PlusOutlined, MinusCircleOutlined } from "@ant-design/icons";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { useCreateProject, ProjectCreateParams } from "@/app/(dashboard)/hooks/projects/useCreateProject";
import { Team } from "../../key_team_helpers/key_list";
import { fetchTeamModels } from "../../organisms/create_key_button";
import { getModelDisplayName } from "../../key_team_helpers/fetch_available_models_team_key";

interface CreateProjectModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function CreateProjectModal({ isOpen, onClose }: CreateProjectModalProps) {
  const [form] = Form.useForm();
  const { accessToken, userId, userRole } = useAuthorized();
  const { data: teams } = useTeams();
  const createMutation = useCreateProject();

  const [selectedTeam, setSelectedTeam] = useState<Team | null>(null);
  const [modelsToPick, setModelsToPick] = useState<string[]>([]);

  // Fetch team-scoped models when team selection changes
  useEffect(() => {
    if (userId && userRole && accessToken && selectedTeam) {
      fetchTeamModels(userId, userRole, accessToken, selectedTeam.team_id).then((models) => {
        const allModels = Array.from(new Set([...(selectedTeam.models ?? []), ...models]));
        setModelsToPick(allModels);
      });
    } else {
      setModelsToPick([]);
    }
    form.setFieldValue("models", []);
  }, [selectedTeam, accessToken, userId, userRole, form]);

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields();

      // Build model-specific limits from the dynamic form list
      const modelRpmLimit: Record<string, number> = {};
      const modelTpmLimit: Record<string, number> = {};
      for (const entry of values.modelLimits ?? []) {
        if (entry.model) {
          if (entry.rpm != null) modelRpmLimit[entry.model] = entry.rpm;
          if (entry.tpm != null) modelTpmLimit[entry.model] = entry.tpm;
        }
      }

      // Build metadata from the dynamic form list
      const metadata: Record<string, unknown> = {};
      for (const entry of values.metadata ?? []) {
        if (entry.key) metadata[entry.key] = entry.value;
      }

      const params: ProjectCreateParams = {
        project_alias: values.project_alias,
        description: values.description,
        team_id: values.team_id,
        models: values.models ?? [],
        max_budget: values.max_budget,
        blocked: values.isBlocked ?? false,
        ...(Object.keys(modelRpmLimit).length > 0 && { model_rpm_limit: modelRpmLimit }),
        ...(Object.keys(modelTpmLimit).length > 0 && { model_tpm_limit: modelTpmLimit }),
        ...(Object.keys(metadata).length > 0 && { metadata }),
      };

      createMutation.mutate(params, {
        onSuccess: () => {
          message.success("Project created successfully");
          form.resetFields();
          setSelectedTeam(null);
          setModelsToPick([]);
          onClose();
        },
        onError: (error) => {
          message.error(error.message || "Failed to create project");
        },
      });
    } catch (error) {
      console.error("Validation failed:", error);
    }
  };

  const handleCancel = () => {
    form.resetFields();
    setSelectedTeam(null);
    setModelsToPick([]);
    onClose();
  };

  const handleTeamChange = (teamId: string) => {
    const team = teams?.find((t) => t.team_id === teamId) ?? null;
    setSelectedTeam(team);
  };

  return (
    <Modal
      title={
        <Typography.Text strong style={{ fontSize: 18 }}>
          Create New Project
        </Typography.Text>
      }
      open={isOpen}
      onCancel={handleCancel}
      width={720}
      footer={[
        <Button key="cancel" onClick={handleCancel}>
          Cancel
        </Button>,
        <Button key="submit" type="primary" icon={<FolderAddOutlined />} loading={createMutation.isPending} onClick={handleSubmit}>
          Create Project
        </Button>,
      ]}
    >
      <Form
        form={form}
        layout="vertical"
        initialValues={{
          isBlocked: false,
        }}
        style={{ marginTop: 24 }}
      >
        {/* Basic Info */}
        <Typography.Text
          strong
          style={{ fontSize: 13, color: "#374151", textTransform: "uppercase", letterSpacing: "0.05em" }}
        >
          Basic Information
        </Typography.Text>
        <Divider style={{ marginTop: 8, marginBottom: 16 }} />

        <Row gutter={24}>
          <Col span={12}>
            <Form.Item
              name="project_alias"
              label="Project Name"
              rules={[{ required: true, message: "Please enter a project name" }]}
            >
              <Input placeholder="e.g. Customer Support Bot" />
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item name="team_id" label="Team" rules={[{ required: true, message: "Please select a team" }]}>
              <Select
                showSearch
                placeholder="Search or select a team"
                onChange={handleTeamChange}
                allowClear
                optionLabelProp="label"
                filterOption={(input, option) => {
                  const team = teams?.find((t) => t.team_id === option?.value);
                  if (!team) return false;
                  const search = input.toLowerCase().trim();
                  return (
                    (team.team_alias || "").toLowerCase().includes(search) ||
                    team.team_id.toLowerCase().includes(search)
                  );
                }}
              >
                {teams?.map((team) => (
                  <Select.Option key={team.team_id} value={team.team_id} label={team.team_alias || team.team_id}>
                    <span style={{ fontWeight: 500 }}>{team.team_alias}</span>{" "}
                    <span style={{ color: "#9ca3af" }}>({team.team_id})</span>
                  </Select.Option>
                ))}
              </Select>
            </Form.Item>
          </Col>
        </Row>

        <Row>
          <Col span={24}>
            <Form.Item name="description" label="Description">
              <Input.TextArea placeholder="Describe the purpose of this project" rows={3} />
            </Form.Item>
          </Col>
        </Row>

        <Row>
          <Col span={24}>
            <Form.Item
              name="models"
              label="Allowed Models (scoped to selected team's models)"
              help={!selectedTeam ? "Select a team first to see available models" : undefined}
            >
              <Select
                mode="multiple"
                placeholder={selectedTeam ? "Select models" : "Select a team first"}
                disabled={!selectedTeam}
                allowClear
                maxTagCount="responsive"
                onChange={(values) => {
                  if (values.includes("all-team-models")) {
                    form.setFieldsValue({ models: ["all-team-models"] });
                  }
                }}
              >
                <Select.Option key="all-team-models" value="all-team-models">
                  All Team Models
                </Select.Option>
                {modelsToPick.map((model) => (
                  <Select.Option key={model} value={model}>
                    {getModelDisplayName(model)}
                  </Select.Option>
                ))}
              </Select>
            </Form.Item>
          </Col>
        </Row>

        <Row gutter={24}>
          <Col span={12}>
            <Form.Item name="max_budget" label="Max Budget (USD)">
              <InputNumber prefix="$" style={{ width: "100%" }} placeholder="0.00" min={0} precision={2} />
            </Form.Item>
          </Col>
        </Row>

        {/* Advanced Settings */}
        <Row>
          <Col span={24}>
            <Collapse ghost style={{ background: "#f9fafb", borderRadius: 8, border: "1px solid #e5e7eb" }}>
              <Collapse.Panel
                header={
                  <Typography.Text strong style={{ color: "#374151" }}>
                    Advanced Settings
                  </Typography.Text>
                }
                key="1"
              >
                <Flex align="center" gap={12}>
                  <Typography.Text strong>Block Project</Typography.Text>
                  <Form.Item name="isBlocked" valuePropName="checked" noStyle>
                    <Switch />
                  </Form.Item>
                </Flex>
                <Form.Item noStyle shouldUpdate={(prev, cur) => prev.isBlocked !== cur.isBlocked}>
                  {({ getFieldValue }) =>
                    getFieldValue("isBlocked") ? (
                      <Alert
                        banner
                        type="warning"
                        showIcon
                        message="All API requests using keys under this project will be rejected."
                        style={{ marginTop: 12 }}
                      />
                    ) : null
                  }
                </Form.Item>

                <Divider />

                <Typography.Text strong style={{ display: "block", marginBottom: 12 }}>
                  Model-Specific Limits
                </Typography.Text>
                <Form.List name="modelLimits">
                  {(fields, { add, remove }) => (
                    <>
                      {fields.map(({ key, name, ...restField }) => (
                        <Space key={key} style={{ display: "flex", marginBottom: 8 }} align="baseline">
                          <Form.Item
                            {...restField}
                            name={[name, "model"]}
                            rules={[{ required: true, message: "Missing model" }]}
                          >
                            <Input placeholder="Model name (e.g. gpt-4)" />
                          </Form.Item>
                          <Form.Item {...restField} name={[name, "tpm"]}>
                            <InputNumber placeholder="TPM Limit" min={0} />
                          </Form.Item>
                          <Form.Item {...restField} name={[name, "rpm"]}>
                            <InputNumber placeholder="RPM Limit" min={0} />
                          </Form.Item>
                          <MinusCircleOutlined onClick={() => remove(name)} style={{ color: "#ef4444" }} />
                        </Space>
                      ))}
                      <Form.Item>
                        <Button type="dashed" onClick={() => add()} block icon={<PlusOutlined />}>
                          Add Model Limit
                        </Button>
                      </Form.Item>
                    </>
                  )}
                </Form.List>

                <Divider />

                <Typography.Text strong style={{ display: "block", marginBottom: 12 }}>
                  Metadata
                </Typography.Text>
                <Form.List name="metadata">
                  {(fields, { add, remove }) => (
                    <>
                      {fields.map(({ key, name, ...restField }) => (
                        <Space key={key} style={{ display: "flex", marginBottom: 8 }} align="baseline">
                          <Form.Item
                            {...restField}
                            name={[name, "key"]}
                            rules={[{ required: true, message: "Missing key" }]}
                          >
                            <Input placeholder="Key" />
                          </Form.Item>
                          <Form.Item
                            {...restField}
                            name={[name, "value"]}
                            rules={[{ required: true, message: "Missing value" }]}
                          >
                            <Input placeholder="Value" />
                          </Form.Item>
                          <MinusCircleOutlined onClick={() => remove(name)} style={{ color: "#ef4444" }} />
                        </Space>
                      ))}
                      <Form.Item>
                        <Button type="dashed" onClick={() => add()} block icon={<PlusOutlined />}>
                          Add Key-Value Pair
                        </Button>
                      </Form.Item>
                    </>
                  )}
                </Form.List>
              </Collapse.Panel>
            </Collapse>
          </Col>
        </Row>
      </Form>
    </Modal>
  );
}
