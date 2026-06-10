import { useEffect, useState } from "react";
import {
  Alert,
  Col,
  Collapse,
  Divider,
  Flex,
  Form,
  Input,
  InputNumber,
  Row,
  Select,
  Space,
  Switch,
  Typography,
  Button,
} from "antd";
import type { FormInstance } from "antd";
import { PlusOutlined, MinusCircleOutlined } from "@ant-design/icons";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useTranslation } from "react-i18next";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { Team } from "../../key_team_helpers/key_list";
import { fetchTeamModels } from "../../organisms/create_key_button";
import { getModelDisplayName } from "../../key_team_helpers/fetch_available_models_team_key";
import { getGuardrailsList } from "@/components/networking";

export interface ProjectFormValues {
  project_alias: string;
  team_id: string;
  description?: string;
  models: string[];
  max_budget?: number;
  isBlocked: boolean;
  guardrails?: string[];
  modelLimits?: { model: string; tpm?: number; rpm?: number }[];
  metadata?: { key: string; value: string }[];
}

interface ProjectBaseFormProps {
  form: FormInstance<ProjectFormValues>;
}

export function ProjectBaseForm({ form }: ProjectBaseFormProps) {
  const { t } = useTranslation();
  const { accessToken, userId, userRole } = useAuthorized();
  const { data: teams } = useTeams();

  const [selectedTeam, setSelectedTeam] = useState<Team | null>(null);
  const [modelsToPick, setModelsToPick] = useState<string[]>([]);
  const [guardrailsList, setGuardrailsList] = useState<string[]>([]);

  useEffect(() => {
    const fetchGuardrails = async () => {
      if (!accessToken) return;
      try {
        const response = await getGuardrailsList(accessToken);
        const names = response.guardrails.map((g: { guardrail_name: string }) => g.guardrail_name);
        setGuardrailsList(names);
      } catch (error) {
        console.error("Failed to fetch guardrails:", error);
      }
    };
    fetchGuardrails();
  }, [accessToken]);

  // Sync selectedTeam from form value (needed for edit mode pre-fill)
  const teamIdValue = Form.useWatch("team_id", form);
  useEffect(() => {
    if (teamIdValue && teams) {
      const team = teams.find((t) => t.team_id === teamIdValue) ?? null;
      if (team && team.team_id !== selectedTeam?.team_id) {
        setSelectedTeam(team);
      }
    }
  }, [teamIdValue, teams, selectedTeam?.team_id]);

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
  }, [selectedTeam, accessToken, userId, userRole]);

  const handleTeamChange = (teamId: string) => {
    const team = teams?.find((t) => t.team_id === teamId) ?? null;
    setSelectedTeam(team);
    form.setFieldValue("models", []);
  };

  return (
    <Form
      form={form}
      layout="vertical"
      name="project_form"
      initialValues={{ isBlocked: false }}
      style={{ marginTop: 24 }}
    >
      {/* Basic Info */}
      <Typography.Text
        strong
        style={{
          fontSize: 13,
          color: "#374151",
          textTransform: "uppercase",
          letterSpacing: "0.05em",
        }}
      >
        {t("projects.projectBaseForm.basicInfoTitle")}
      </Typography.Text>
      <Divider style={{ marginTop: 8, marginBottom: 16 }} />

      <Row gutter={24}>
        <Col span={12}>
          <Form.Item
            name="project_alias"
            label={t("projects.projectBaseForm.projectNameLabel")}
            rules={[{ required: true, message: t("projects.projectBaseForm.projectNameRequired") }]}
          >
            <Input placeholder={t("projects.projectBaseForm.projectNamePlaceholder")} />
          </Form.Item>
        </Col>
        <Col span={12}>
          <Form.Item
            name="team_id"
            label={t("projects.projectBaseForm.teamLabel")}
            rules={[{ required: true, message: t("projects.projectBaseForm.teamRequired") }]}
          >
            <Select
              showSearch
              placeholder={t("projects.projectBaseForm.teamPlaceholder")}
              onChange={handleTeamChange}
              allowClear
              optionLabelProp="label"
              filterOption={(input, option) => {
                const team = teams?.find((t) => t.team_id === option?.value);
                if (!team) return false;
                const search = input.toLowerCase().trim();
                return (
                  (team.team_alias || "").toLowerCase().includes(search) || team.team_id.toLowerCase().includes(search)
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
          <Form.Item name="description" label={t("common.description")}>
            <Input.TextArea placeholder={t("projects.projectBaseForm.descriptionPlaceholder")} rows={3} />
          </Form.Item>
        </Col>
      </Row>

      <Row>
        <Col span={24}>
          <Form.Item
            name="models"
            label={t("projects.projectBaseForm.allowedModelsLabel")}
            help={!selectedTeam ? t("projects.projectBaseForm.selectTeamFirst") : undefined}
          >
            <Select
              mode="multiple"
              placeholder={
                selectedTeam
                  ? t("projects.projectBaseForm.selectModels")
                  : t("projects.projectBaseForm.selectTeamFirstPlaceholder")
              }
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
                {t("projects.projectBaseForm.allTeamModels")}
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
          <Form.Item name="max_budget" label={t("projects.projectBaseForm.maxBudgetLabel")}>
            <InputNumber prefix="$" style={{ width: "100%" }} placeholder="0.00" min={0} precision={2} />
          </Form.Item>
        </Col>
      </Row>

      {/* Advanced Settings */}
      <Row>
        <Col span={24}>
          <Collapse
            ghost
            style={{
              background: "#f9fafb",
              borderRadius: 8,
              border: "1px solid #e5e7eb",
            }}
            items={[
              {
                key: "1",
                label: (
                  <Typography.Text strong style={{ color: "#374151" }}>
                    {t("projects.projectBaseForm.advancedSettingsTitle")}
                  </Typography.Text>
                ),
                children: (
                  <>
                    <Flex align="center" gap={12}>
                      <Typography.Text strong>{t("projects.projectBaseForm.blockProjectLabel")}</Typography.Text>
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
                            message={t("projects.projectBaseForm.blockProjectWarning")}
                            style={{ marginTop: 12 }}
                          />
                        ) : null
                      }
                    </Form.Item>

                    <Divider />

                    <Form.Item
                      label={t("projects.projectBaseForm.guardrailsLabel")}
                      name="guardrails"
                      help={t("projects.projectBaseForm.guardrailsHelp")}
                    >
                      <Select
                        mode="tags"
                        style={{ width: "100%" }}
                        placeholder={t("projects.projectBaseForm.guardrailsPlaceholder")}
                        options={guardrailsList.map((name) => ({
                          value: name,
                          label: name,
                        }))}
                      />
                    </Form.Item>

                    <Divider />

                    <Typography.Text strong style={{ display: "block", marginBottom: 12 }}>
                      {t("projects.projectBaseForm.modelLimitsTitle")}
                    </Typography.Text>
                    <Form.List name="modelLimits">
                      {(fields, { add, remove }) => (
                        <>
                          {fields.map(({ key, name, ...restField }) => (
                            <Space key={key} style={{ display: "flex", marginBottom: 8 }} align="baseline">
                              <Form.Item
                                {...restField}
                                name={[name, "model"]}
                                rules={[
                                  { required: true, message: t("projects.projectBaseForm.missingModel") },
                                  {
                                    validator: (_, value) => {
                                      if (!value) return Promise.resolve();
                                      const all = form.getFieldValue("modelLimits") ?? [];
                                      const dupes = all.filter((entry: { model?: string }) => entry?.model === value);
                                      if (dupes.length > 1) {
                                        return Promise.reject(new Error(t("projects.projectBaseForm.duplicateModel")));
                                      }
                                      return Promise.resolve();
                                    },
                                  },
                                ]}
                              >
                                <Input placeholder={t("projects.projectBaseForm.modelNamePlaceholder")} />
                              </Form.Item>
                              <Form.Item {...restField} name={[name, "tpm"]}>
                                <InputNumber placeholder={t("projects.projectBaseForm.tpmPlaceholder")} min={0} />
                              </Form.Item>
                              <Form.Item {...restField} name={[name, "rpm"]}>
                                <InputNumber placeholder={t("projects.projectBaseForm.rpmPlaceholder")} min={0} />
                              </Form.Item>
                              <MinusCircleOutlined onClick={() => remove(name)} style={{ color: "#ef4444" }} />
                            </Space>
                          ))}
                          <Form.Item>
                            <Button type="dashed" onClick={() => add()} block icon={<PlusOutlined />}>
                              {t("projects.projectBaseForm.addModelLimit")}
                            </Button>
                          </Form.Item>
                        </>
                      )}
                    </Form.List>

                    <Divider />

                    <Typography.Text strong style={{ display: "block", marginBottom: 12 }}>
                      {t("projects.projectBaseForm.metadataTitle")}
                    </Typography.Text>
                    <Form.List name="metadata">
                      {(fields, { add, remove }) => (
                        <>
                          {fields.map(({ key, name, ...restField }) => (
                            <Space key={key} style={{ display: "flex", marginBottom: 8 }} align="baseline">
                              <Form.Item
                                {...restField}
                                name={[name, "key"]}
                                rules={[
                                  { required: true, message: t("projects.projectBaseForm.missingKey") },
                                  {
                                    validator: (_, value) => {
                                      if (!value) return Promise.resolve();
                                      const all = form.getFieldValue("metadata") ?? [];
                                      const dupes = all.filter((entry: { key?: string }) => entry?.key === value);
                                      if (dupes.length > 1) {
                                        return Promise.reject(new Error(t("projects.projectBaseForm.duplicateKey")));
                                      }
                                      return Promise.resolve();
                                    },
                                  },
                                ]}
                              >
                                <Input placeholder={t("projects.projectBaseForm.keyPlaceholder")} />
                              </Form.Item>
                              <Form.Item
                                {...restField}
                                name={[name, "value"]}
                                rules={[{ required: true, message: t("projects.projectBaseForm.missingValue") }]}
                              >
                                <Input placeholder={t("projects.projectBaseForm.valuePlaceholder")} />
                              </Form.Item>
                              <MinusCircleOutlined onClick={() => remove(name)} style={{ color: "#ef4444" }} />
                            </Space>
                          ))}
                          <Form.Item>
                            <Button type="dashed" onClick={() => add()} block icon={<PlusOutlined />}>
                              {t("projects.projectBaseForm.addKeyValue")}
                            </Button>
                          </Form.Item>
                        </>
                      )}
                    </Form.List>
                  </>
                ),
              },
            ]}
          />
        </Col>
      </Row>
    </Form>
  );
}
