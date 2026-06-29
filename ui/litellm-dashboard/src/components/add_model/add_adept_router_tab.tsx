import React, { useEffect, useState } from "react";
import {
  Alert,
  Button,
  Card,
  Collapse,
  Form,
  Input,
  InputNumber,
  Select as AntdSelect,
  Tag,
  Tooltip,
  Typography,
} from "antd";
import type { FormInstance } from "antd";
import { modelAvailableCall } from "../networking";
import { all_admin_roles } from "@/utils/roles";
import { handleAddAdeptRouterSubmit } from "./handle_add_adept_router_submit";
import { fetchAvailableModels, ModelGroup } from "../llm_calls/fetch_models";
import NotificationManager from "../molecules/notifications_manager";

interface AddAdeptRouterTabProps {
  form: FormInstance;
  // Refresh-only callback invoked after the ADEPT router is created. Must NOT
  // be the regular Add Model form's submit handler — otherwise it triggers
  // empty-form validation errors after a successful save.
  onSuccess?: () => void;
  accessToken: string;
  userRole: string;
}

const { Title, Text, Paragraph } = Typography;

const TRAINER_SQL_EXAMPLE = `-- Read training data for a template
SELECT prompt, response, additional_information
FROM conversations
WHERE template_id = '<template_id>';

-- Signal completion: update target_model so the router uses your SLM
UPDATE templates
SET target_model = 'your-trained-model-name'
WHERE id = '<template_id>';`;

const AddAdeptRouterTab: React.FC<AddAdeptRouterTabProps> = ({
  form,
  onSuccess,
  accessToken,
  userRole,
}) => {
  const [modelAccessGroups, setModelAccessGroups] = useState<string[]>([]);
  const [modelInfo, setModelInfo] = useState<ModelGroup[]>([]);
  const [tagPrefix, setTagPrefix] = useState<string>("");

  useEffect(() => {
    const fetchModelAccessGroups = async () => {
      const response = await modelAvailableCall(accessToken, "", "", false, null, true, true);
      setModelAccessGroups(response["data"].map((model: any) => model["id"]));
    };
    fetchModelAccessGroups();
  }, [accessToken]);

  useEffect(() => {
    const loadModels = async () => {
      try {
        setModelInfo(await fetchAvailableModels(accessToken));
      } catch {
        // model list unavailable
      }
    };
    loadModels();
  }, [accessToken]);

  const isAdmin = all_admin_roles.includes(userRole);

  const modelOptions = Array.from(new Set(modelInfo.map((m) => m.model_group))).map((g) => ({
    value: g,
    label: g,
  }));

  const handleSubmit = () => {
    const currentValues = form.getFieldsValue();

    if (!currentValues.adept_router_name) {
      NotificationManager.fromBackend("Please enter a Router Name");
      return;
    }

    if (!currentValues.adept_router_default_model) {
      NotificationManager.fromBackend("Please select a Default Model");
      return;
    }

    form
      .validateFields()
      .then((values) => {
        handleAddAdeptRouterSubmit(values, accessToken, form, onSuccess);
      })
      .catch(() => {
        NotificationManager.fromBackend("Please fill in all required fields");
      });
  };

  const tagPrefixPreview = tagPrefix
    ? `<${tagPrefix}_fieldname>value</${tagPrefix}_fieldname>`
    : null;

  return (
    <>
      <Title level={2}>Add ADEPT Router</Title>
      <Text className="text-gray-600 mb-6">
        ADEPT routing learns from your traffic. It extracts a structural template from each
        prompt by masking variable content, then matches future requests to known templates.
      </Text>

      <Card className="mb-4">
        <Alert
          type="info"
          showIcon
          className="mb-4"
          message="How ADEPT Routing works"
          description={
            <div>
              <Paragraph className="mb-2">
                ADEPT extracts a <strong>template</strong> from each prompt by masking variable
                spans (numbers, emails, URLs, IDs). Structurally equivalent prompts produce the
                same template and are routed to the same model.
              </Paragraph>
              <Paragraph className="mb-2">
                If you set a <strong>Tag Prefix</strong>, wrap variable parts of your prompts in
                XML tags — ADEPT strips those values before matching:
              </Paragraph>
              <pre className="bg-gray-100 p-2 rounded text-xs overflow-x-auto">
                {`Tag prefix: "var"\n\nPrompt:\n  Get order <var_id>ORD-123</var_id> for <var_email>user@acme.com</var_email>\n\nTemplate (after extraction):\n  Get order <var_id></var_id> for <var_email></var_email>`}
              </pre>
              <Paragraph className="mt-2 mb-0">
                Without tags, automatic masking handles numbers, emails, URLs, and UUIDs.
              </Paragraph>
            </div>
          }
        />

        <Form form={form} labelCol={{ span: 10 }} wrapperCol={{ span: 16 }} labelAlign="left">
          <Form.Item
            rules={[{ required: true, message: "Router name is required" }]}
            label="Router Name"
            name="adept_router_name"
            tooltip="Unique name for this ADEPT router"
          >
            <Input placeholder="e.g., adept_router_prod" />
          </Form.Item>

          <Form.Item
            rules={[{ required: true, message: "Default model is required" }]}
            label="Default Model"
            name="adept_router_default_model"
            tooltip="Fallback model used when no template matches"
          >
            <AntdSelect
              placeholder="Select a default model"
              options={modelOptions}
              style={{ width: "100%" }}
              showSearch
            />
          </Form.Item>

          <Form.Item
            label="Tag Prefix"
            name="adept_router_tag_prefix"
            tooltip="Optional XML tag prefix users add around variable content in their prompts"
          >
            <Input placeholder="e.g., var" onChange={(e) => setTagPrefix(e.target.value)} />
          </Form.Item>
          {tagPrefixPreview && (
            <Form.Item wrapperCol={{ offset: 10, span: 16 }}>
              <Text type="secondary" className="text-xs">
                Tags will look like: <code>{tagPrefixPreview}</code>
              </Text>
            </Form.Item>
          )}

          <div className="flex items-center my-4">
            <div className="flex-grow border-t border-gray-200" />
            <span className="px-4 text-gray-500 text-sm">Database</span>
            <div className="flex-grow border-t border-gray-200" />
          </div>

          <Form.Item
            label="Host"
            name="adept_router_pg_host"
            rules={[{ required: true, message: "Host is required" }]}
            tooltip="PostgreSQL server hostname or IP"
          >
            <Input placeholder="e.g., db.internal.com" />
          </Form.Item>
          <Form.Item label="Port" name="adept_router_pg_port" initialValue={5432}>
            <InputNumber min={1} max={65535} style={{ width: "100%" }} />
          </Form.Item>
          <Form.Item
            label="Database"
            name="adept_router_pg_database"
            rules={[{ required: true, message: "Database name is required" }]}
          >
            <Input placeholder="e.g., adept_db" />
          </Form.Item>
          <Form.Item
            label="Username"
            name="adept_router_pg_user"
            rules={[{ required: true, message: "Username is required" }]}
          >
            <Input placeholder="e.g., adept_user" />
          </Form.Item>
          <Form.Item
            label="Password"
            name="adept_router_pg_password"
            rules={[{ required: true, message: "Password is required" }]}
            tooltip="Stored encrypted at rest"
          >
            <Input.Password placeholder="Database password" />
          </Form.Item>

          <div className="flex items-center my-4">
            <div className="flex-grow border-t border-gray-200" />
            <span className="px-4 text-gray-500 text-sm">Additional Settings</span>
            <div className="flex-grow border-t border-gray-200" />
          </div>

          <Form.Item
            label="Conversations Threshold"
            name="adept_router_conversations_threshold"
            initialValue={10}
            tooltip="Number of conversations per template after which the trainer is notified. Re-triggers at every multiple of this number."
          >
            <InputNumber min={1} style={{ width: "100%" }} />
          </Form.Item>

          <Collapse
            ghost
            items={[
              {
                key: "trainer",
                label: (
                  <Text strong>
                    Trainer Integration{" "}
                    <Tag color="orange" className="font-normal">Experimental</Tag>
                  </Text>
                ),
                children: (
                  <div>
                    <Paragraph type="secondary" className="text-sm mb-3">
                      Connect your own training pipeline. When a template reaches the
                      conversations threshold, ADEPT calls your trainer URL. Your pipeline
                      reads training data from the database and writes back the trained model name.
                    </Paragraph>
                    <Form.Item
                      label="Trainer URL"
                      name="adept_router_trainer_url"
                      tooltip="Optional HTTP endpoint called when the threshold is reached"
                    >
                      <Input placeholder="e.g., https://trainer.internal.com" />
                    </Form.Item>
                    <Alert
                      type="info"
                      showIcon
                      className="mt-2"
                      message="Integration contract"
                      description={
                        <div>
                          <Paragraph className="text-xs mb-1">
                            <strong>Trigger:</strong>{" "}
                            <code>POST {"{trainer_url}"}/run-workflow/{"{template_id}"}</code>
                            {" "}fires at every N conversations (N = threshold above).
                          </Paragraph>
                          <Paragraph className="text-xs mb-1">
                            <strong>Reading data &amp; completing the loop:</strong>
                          </Paragraph>
                          <pre className="bg-gray-50 p-2 rounded text-xs overflow-x-auto mb-1">
                            {TRAINER_SQL_EXAMPLE}
                          </pre>
                          <Paragraph className="text-xs mb-0">
                            <strong>Registering the model:</strong> add the trained model as a
                            regular LiteLLM model entry so ADEPT can route to it. The router
                            picks up the updated <code>target_model</code> on the next request
                            with no restart needed.
                          </Paragraph>
                        </div>
                      }
                    />
                  </div>
                ),
              },
            ]}
          />

          {isAdmin && (
            <Form.Item
              label="Model Access Group"
              name="model_access_group"
              tooltip="Control which teams can access this ADEPT router"
              className="mt-4"
            >
              <AntdSelect
                mode="tags"
                showSearch
                placeholder="Select existing groups or type to create new ones"
                tokenSeparators={[","]}
                options={modelAccessGroups.map((g) => ({ value: g, label: g }))}
                maxTagCount="responsive"
                allowClear
              />
            </Form.Item>
          )}

          <div className="flex justify-between items-center mt-4">
            <Tooltip title="Get help on our GitHub">
              <Typography.Link href="https://github.com/BerriAI/litellm/issues">
                Need Help?
              </Typography.Link>
            </Tooltip>
            <Button type="primary" onClick={handleSubmit}>
              Add ADEPT Router
            </Button>
          </div>
        </Form>
      </Card>
    </>
  );
};

export default AddAdeptRouterTab;
