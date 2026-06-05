import React, { useEffect, useState } from "react";
import {
  Alert,
  Button,
  Collapse,
  Form,
  Input,
  InputNumber,
  Modal,
  Select as AntdSelect,
  Tag,
  Typography,
} from "antd";
import { modelAvailableCall, modelPatchUpdateCall } from "../networking";
import { fetchAvailableModels, ModelGroup } from "../playground/llm_calls/fetch_models";
import NotificationsManager from "../molecules/notifications_manager";
import { all_admin_roles } from "@/utils/roles";

interface EditAdeptRouterModalProps {
  isVisible: boolean;
  onCancel: () => void;
  onSuccess: (updatedModel: any) => void;
  modelData: any;
  accessToken: string;
  userRole: string;
}

const { Text, Paragraph } = Typography;

const TRAINER_SQL_EXAMPLE = `-- Read training data for a template
SELECT prompt, response, additional_information
FROM conversations
WHERE template_id = '<template_id>';

-- Signal completion: update target_model so the router uses your SLM
UPDATE templates
SET target_model = 'your-trained-model-name'
WHERE id = '<template_id>';`;

const EditAdeptRouterModal: React.FC<EditAdeptRouterModalProps> = ({
  isVisible,
  onCancel,
  onSuccess,
  modelData,
  accessToken,
  userRole,
}) => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [modelAccessGroups, setModelAccessGroups] = useState<string[]>([]);
  const [modelInfo, setModelInfo] = useState<ModelGroup[]>([]);
  const [tagPrefix, setTagPrefix] = useState<string>("");
  const [pgPasswordAlreadySet, setPgPasswordAlreadySet] = useState(false);

  const isAdmin = all_admin_roles.includes(userRole);

  useEffect(() => {
    if (!isVisible) return;

    const fetchModelAccessGroups = async () => {
      try {
        const response = await modelAvailableCall(accessToken, "", "", false, null, true, true);
        setModelAccessGroups(response["data"].map((m: any) => m["id"]));
      } catch {
        // ignore
      }
    };

    const loadModels = async () => {
      try {
        setModelInfo(await fetchAvailableModels(accessToken));
      } catch {
        // ignore
      }
    };

    fetchModelAccessGroups();
    loadModels();
  }, [isVisible, accessToken]);

  useEffect(() => {
    if (!isVisible || !modelData) return;

    const lp = modelData.litellm_params || {};
    const prefix = lp.adept_router_tag_prefix || "";

    setTagPrefix(prefix);
    setPgPasswordAlreadySet(!!lp.adept_router_pg_password);

    form.setFieldsValue({
      adept_router_name: modelData.model_name,
      adept_router_default_model: lp.adept_router_default_model || "",
      adept_router_tag_prefix: prefix,
      adept_router_conversations_threshold: lp.adept_router_conversations_threshold || 10,
      adept_router_trainer_url: lp.adept_router_trainer_url || "",
      adept_router_pg_host: lp.adept_router_pg_host || "",
      adept_router_pg_port: lp.adept_router_pg_port || 5432,
      adept_router_pg_database: lp.adept_router_pg_database || "",
      adept_router_pg_user: lp.adept_router_pg_user || "",
      // password intentionally not pre-populated
      model_access_group: modelData.model_info?.access_groups || [],
    });
  }, [isVisible, modelData, form]);

  const modelOptions = Array.from(new Set(modelInfo.map((m) => m.model_group))).map((g) => ({
    value: g,
    label: g,
  }));

  const handleSubmit = async () => {
    try {
      setLoading(true);
      const values = await form.validateFields();
      const updatedLitellmParams: Record<string, any> = {
        ...modelData.litellm_params,
        adept_router_default_model: values.adept_router_default_model,
        adept_router_tag_prefix: values.adept_router_tag_prefix || undefined,
        adept_router_conversations_threshold: values.adept_router_conversations_threshold || undefined,
        adept_router_trainer_url: values.adept_router_trainer_url || undefined,
        adept_router_pg_host: values.adept_router_pg_host || undefined,
        adept_router_pg_port: values.adept_router_pg_port || undefined,
        adept_router_pg_database: values.adept_router_pg_database || undefined,
        adept_router_pg_user: values.adept_router_pg_user || undefined,
      };

      if (values.adept_router_pg_password) {
        updatedLitellmParams.adept_router_pg_password = values.adept_router_pg_password;
      }

      const updatedModelInfo = {
        ...modelData.model_info,
        access_groups: values.model_access_group || [],
      };

      const updatePayload = {
        model_name: values.adept_router_name,
        litellm_params: updatedLitellmParams,
        model_info: updatedModelInfo,
      };

      await modelPatchUpdateCall(accessToken, updatePayload, modelData.model_info.id);

      NotificationsManager.success("ADEPT router updated successfully");
      onSuccess({ ...modelData, ...updatePayload });
      onCancel();
    } catch (error) {
      NotificationsManager.fromBackend("Failed to update ADEPT router: " + error);
    } finally {
      setLoading(false);
    }
  };

  const tagPrefixPreview = tagPrefix
    ? `<${tagPrefix}_fieldname>value</${tagPrefix}_fieldname>`
    : null;

  return (
    <Modal
      title="Edit ADEPT Router"
      open={isVisible}
      onCancel={onCancel}
      width={800}
      destroyOnHidden
      footer={[
        <Button key="cancel" onClick={onCancel}>
          Cancel
        </Button>,
        <Button key="submit" type="primary" loading={loading} onClick={handleSubmit}>
          Save Changes
        </Button>,
      ]}
    >
      <Form form={form} labelCol={{ span: 10 }} wrapperCol={{ span: 14 }} labelAlign="left">
        <Form.Item
          label="Router Name"
          name="adept_router_name"
          rules={[{ required: true, message: "Router name is required" }]}
        >
          <Input disabled />
        </Form.Item>

        <Form.Item
          label="Default Model"
          name="adept_router_default_model"
          rules={[{ required: true, message: "Default model is required" }]}
        >
          <AntdSelect options={modelOptions} showSearch style={{ width: "100%" }} />
        </Form.Item>

        <Form.Item
          label="Tag Prefix"
          name="adept_router_tag_prefix"
          tooltip="Optional XML tag prefix users wrap variable content with"
        >
          <Input
            placeholder="e.g., var"
            onChange={(e) => setTagPrefix(e.target.value)}
          />
        </Form.Item>
        {tagPrefixPreview && (
          <Form.Item wrapperCol={{ offset: 10, span: 14 }}>
            <Text type="secondary" className="text-xs">
              Tags: <code>{tagPrefixPreview}</code>
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
        <Form.Item label="Port" name="adept_router_pg_port">
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
          tooltip="Stored encrypted at rest. Leave blank to keep the current password."
        >
          <Input.Password
            placeholder={pgPasswordAlreadySet ? "•••••••• (leave blank to keep)" : "Database password"}
          />
        </Form.Item>

        <div className="flex items-center my-4">
          <div className="flex-grow border-t border-gray-200" />
          <span className="px-4 text-gray-500 text-sm">Additional Settings</span>
          <div className="flex-grow border-t border-gray-200" />
        </div>

        <Form.Item
          label="Conversations Threshold"
          name="adept_router_conversations_threshold"
          tooltip="Trainer is notified at every multiple of this number of conversations per template."
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
                    Connect your own training pipeline. When a template reaches the conversations
                    threshold, ADEPT calls your trainer URL.
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
                          {" "}fires at every N conversations (N = threshold).
                        </Paragraph>
                        <Paragraph className="text-xs mb-1">
                          <strong>Reading data &amp; completing the loop:</strong>
                        </Paragraph>
                        <pre className="bg-gray-50 p-2 rounded text-xs overflow-x-auto mb-1">
                          {TRAINER_SQL_EXAMPLE}
                        </pre>
                        <Paragraph className="text-xs mb-0">
                          <strong>Registering the model:</strong> add the trained model as a
                          regular LiteLLM model entry. The router picks up the updated{" "}
                          <code>target_model</code> on the next request — no restart needed.
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
      </Form>
    </Modal>
  );
};

export default EditAdeptRouterModal;
