"use client";

import { useMCPSemanticFilterSettings } from "@/app/(dashboard)/hooks/mcpSemanticFilterSettings/useMCPSemanticFilterSettings";
import { useUpdateMCPSemanticFilterSettings } from "@/app/(dashboard)/hooks/mcpSemanticFilterSettings/useUpdateMCPSemanticFilterSettings";
import NotificationManager from "@/components/molecules/notifications_manager";
import {
  Alert,
  Button,
  Card,
  Col,
  Form,
  InputNumber,
  Row,
  Select,
  Skeleton,
  Slider,
  Space,
  Switch,
  Typography,
  Tooltip,
} from "antd";
import { QuestionCircleOutlined, CheckCircleOutlined, SaveOutlined } from "@ant-design/icons";
import { useEffect, useState } from "react";
import { fetchAvailableModels, ModelGroup } from "@/components/playground/llm_calls/fetch_models";
import MCPSemanticFilterTestPanel from "./MCPSemanticFilterTestPanel";
import { getCurlCommand, runSemanticFilterTest, TestResult } from "./semanticFilterTestUtils";

interface MCPSemanticFilterSettingsProps {
  accessToken: string | null;
}

export default function MCPSemanticFilterSettings({ accessToken }: MCPSemanticFilterSettingsProps) {
  const { data, isLoading, isError, error } = useMCPSemanticFilterSettings();
  const {
    mutate: updateSettings,
    isPending: isUpdating,
    error: updateError,
  } = useUpdateMCPSemanticFilterSettings(accessToken || "");
  const [form] = Form.useForm();
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [isDirty, setIsDirty] = useState(false);
  const [embeddingModels, setEmbeddingModels] = useState<ModelGroup[]>([]);
  const [loadingModels, setLoadingModels] = useState(true);

  // Test section state
  const [testQuery, setTestQuery] = useState("");
  const [testModel, setTestModel] = useState<string>("gpt-4o");
  const [testResult, setTestResult] = useState<TestResult | null>(null);
  const [isTesting, setIsTesting] = useState(false);

  const schema = data?.field_schema;
  const values = data?.values ?? {};

  useEffect(() => {
    const loadEmbeddingModels = async () => {
      if (!accessToken) return;
      try {
        setLoadingModels(true);
        const models = await fetchAvailableModels(accessToken);
        const embeddingOnly = models.filter((model) => model.mode === "embedding");
        setEmbeddingModels(embeddingOnly);
      } catch (error) {
        console.error("Error fetching embedding models:", error);
      } finally {
        setLoadingModels(false);
      }
    };

    loadEmbeddingModels();
  }, [accessToken]);

  useEffect(() => {
    if (values) {
      form.setFieldsValue({
        enabled: values.enabled ?? false,
        embedding_model: values.embedding_model ?? "text-embedding-3-small",
        top_k: values.top_k ?? 10,
        similarity_threshold: values.similarity_threshold ?? 0.3,
      });
      setIsDirty(false);
    }
  }, [values, form]);

  const handleSave = async () => {
    try {
      const formValues = await form.validateFields();
      updateSettings(formValues, {
        onSuccess: () => {
          setIsDirty(false);
          setSaveSuccess(true);
          setTimeout(() => setSaveSuccess(false), 3000);
          NotificationManager.success(
            "Settings updated successfully. Changes will be applied across all pods within 10 seconds."
          );
        },
        onError: (error) => {
          NotificationManager.fromBackend(error);
        },
      });
    } catch (error) {
      console.error("Form validation failed:", error);
    }
  };

  const handleTest = async () => {
    if (!accessToken) {
      return;
    }

    await runSemanticFilterTest({
      accessToken,
      testModel,
      testQuery,
      setIsTesting,
      setTestResult,
    });
  };

  if (!accessToken) {
    return (
      <div className="p-6 text-center text-gray-500">
        Please log in to configure semantic filter settings.
      </div>
    );
  }

  return (
    <div style={{ width: "100%" }}>
      {isLoading ? (
        <Skeleton active />
      ) : isError ? (
        <Alert
          type="error"
          message="Could not load MCP Semantic Filter settings"
          description={error instanceof Error ? error.message : undefined}
          style={{ marginBottom: 24 }}
        />
      ) : (
        <>
          <Alert
            type="info"
            message="Semantic Tool Filtering"
            description="Filter MCP tools semantically based on query relevance. This reduces context window size and improves tool selection accuracy. Click 'Save Settings' to apply changes across all pods (takes effect within 10 seconds)."
            showIcon
            style={{ marginBottom: 24 }}
          />

          {saveSuccess && (
            <Alert
              type="success"
              message="Settings saved successfully"
              icon={<CheckCircleOutlined />}
              showIcon
              closable
              style={{ marginBottom: 16 }}
            />
          )}

          {updateError && (
            <Alert
              type="error"
              message="Could not update settings"
              description={
                updateError instanceof Error ? updateError.message : undefined
              }
              style={{ marginBottom: 16 }}
            />
          )}

          <Row gutter={24}>
            {/* Left Column - Settings */}
            <Col xs={24} lg={12}>
              <Form
                form={form}
                layout="vertical"
                disabled={isUpdating}
                onValuesChange={() => {
                  setIsDirty(true);
                }}
              >
                <Card style={{ marginBottom: 16 }}>
                  <Form.Item
                    name="enabled"
                    label={
                      <Space>
                        <Typography.Text strong>Enable Semantic Filtering</Typography.Text>
                        <Tooltip title="When enabled, only the most relevant MCP tools will be included in requests based on semantic similarity">
                          <QuestionCircleOutlined style={{ color: "#8c8c8c" }} />
                        </Tooltip>
                      </Space>
                    }
                    valuePropName="checked"
                  >
                    <Switch disabled={isUpdating} />
                  </Form.Item>

                  <Typography.Text type="secondary" style={{ display: "block", marginTop: -16, marginBottom: 16 }}>
                    {schema?.properties?.enabled?.description}
                  </Typography.Text>
                </Card>

                <Card title="Configuration" style={{ marginBottom: 16 }}>
                  <Form.Item
                    name="embedding_model"
                    label={
                      <Space>
                        <Typography.Text strong>Embedding Model</Typography.Text>
                        <Tooltip title="The model used to generate embeddings for semantic matching">
                          <QuestionCircleOutlined style={{ color: "#8c8c8c" }} />
                        </Tooltip>
                      </Space>
                    }
                  >
                    <Select
                      options={embeddingModels.map((model) => ({
                        label: model.model_group,
                        value: model.model_group,
                      }))}
                      placeholder={loadingModels ? "Loading models..." : "Select embedding model"}
                      showSearch
                      disabled={isUpdating || loadingModels}
                      loading={loadingModels}
                      notFoundContent={
                        loadingModels ? "Loading..." : "No embedding models available"
                      }
                    />
                  </Form.Item>

                  <Form.Item
                    name="top_k"
                    label={
                      <Space>
                        <Typography.Text strong>Top K Results</Typography.Text>
                        <Tooltip title="Maximum number of tools to return after filtering">
                          <QuestionCircleOutlined style={{ color: "#8c8c8c" }} />
                        </Tooltip>
                      </Space>
                    }
                  >
                    <InputNumber
                      min={1}
                      max={100}
                      style={{ width: "100%" }}
                      disabled={isUpdating}
                    />
                  </Form.Item>

                  <Form.Item
                    name="similarity_threshold"
                    label={
                      <Space>
                        <Typography.Text strong>Similarity Threshold</Typography.Text>
                        <Tooltip title="Minimum similarity score (0-1) for a tool to be included">
                          <QuestionCircleOutlined style={{ color: "#8c8c8c" }} />
                        </Tooltip>
                      </Space>
                    }
                  >
                    <Slider
                      min={0}
                      max={1}
                      step={0.05}
                      marks={{
                        0: "0.0",
                        0.3: "0.3",
                        0.5: "0.5",
                        0.7: "0.7",
                        1: "1.0",
                      }}
                      disabled={isUpdating}
                    />
                  </Form.Item>
                </Card>

                <div style={{ display: "flex", justifyContent: "flex-end", gap: 8 }}>
                  <Button
                    type="primary"
                    icon={<SaveOutlined />}
                    onClick={handleSave}
                    loading={isUpdating}
                    disabled={!isDirty}
                  >
                    Save Settings
                  </Button>
                </div>
              </Form>
            </Col>

            {/* Right Column - Test Configuration */}
            <Col xs={24} lg={12}>
              <MCPSemanticFilterTestPanel
                accessToken={accessToken}
                testQuery={testQuery}
                setTestQuery={setTestQuery}
                testModel={testModel}
                setTestModel={setTestModel}
                isTesting={isTesting}
                onTest={handleTest}
                filterEnabled={!!values.enabled}
                testResult={testResult}
                curlCommand={getCurlCommand(testModel, testQuery)}
              />
            </Col>
          </Row>
        </>
      )}
    </div>
  );
}
