import React, { useState, useEffect } from "react";
import { Modal, Form, Input, Button as AntButton, message, Switch, Collapse } from "antd";
import { PlusOutlined, MinusCircleOutlined } from "@ant-design/icons";
import { createAgentCall } from "../networking";

const { Panel } = Collapse;

interface AddAgentFormProps {
  visible: boolean;
  onClose: () => void;
  accessToken: string | null;
  onSuccess: () => void;
}

const AddAgentForm: React.FC<AddAgentFormProps> = ({
  visible,
  onClose,
  accessToken,
  onSuccess,
}) => {
  const [form] = Form.useForm();
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (values: any) => {
    if (!accessToken) {
      message.error("No access token available");
      return;
    }

    setIsSubmitting(true);
    try {
      // Build the agent data according to AgentConfig spec
      const agentData = {
        agent_name: values.agent_name,
        agent_card_params: {
          protocolVersion: values.protocolVersion || "1.0",
          name: values.name,
          description: values.description,
          url: values.url,
          version: values.version || "1.0.0",
          defaultInputModes: values.defaultInputModes || ["text"],
          defaultOutputModes: values.defaultOutputModes || ["text"],
          capabilities: {
            streaming: values.streaming !== false,
            ...(values.pushNotifications !== undefined && { pushNotifications: values.pushNotifications }),
            ...(values.stateTransitionHistory !== undefined && { stateTransitionHistory: values.stateTransitionHistory }),
          },
          skills: values.skills || [],
          ...(values.iconUrl && { iconUrl: values.iconUrl }),
          ...(values.documentationUrl && { documentationUrl: values.documentationUrl }),
          ...(values.supportsAuthenticatedExtendedCard !== undefined && { 
            supportsAuthenticatedExtendedCard: values.supportsAuthenticatedExtendedCard 
          }),
        },
        ...(values.model || values.make_public !== undefined ? {
          litellm_params: {
            ...(values.model && { model: values.model }),
            ...(values.make_public !== undefined && { make_public: values.make_public }),
          }
        } : {}),
      };

      await createAgentCall(accessToken, agentData);
      message.success("Agent created successfully");
      form.resetFields();
      onSuccess();
      onClose();
    } catch (error) {
      console.error("Error creating agent:", error);
      message.error("Failed to create agent");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCancel = () => {
    form.resetFields();
    onClose();
  };

  return (
    <Modal
      title="Add New Agent"
      open={visible}
      onCancel={handleCancel}
      footer={null}
      width={800}
    >
      <Form
        form={form}
        layout="vertical"
        onFinish={handleSubmit}
        initialValues={{
          protocolVersion: "1.0",
          version: "1.0.0",
          defaultInputModes: ["text"],
          defaultOutputModes: ["text"],
          streaming: true,
          skills: [],
        }}
      >
        <Form.Item
          label="Agent Name"
          name="agent_name"
          rules={[{ required: true, message: "Please enter a unique agent name" }]}
          tooltip="Unique identifier for the agent"
        >
          <Input placeholder="e.g., customer-support-agent" />
        </Form.Item>

        <Collapse defaultActiveKey={['basic']} style={{ marginBottom: 16 }}>
          <Panel header="Basic Information (Required)" key="basic">
            <Form.Item
              label="Display Name"
              name="name"
              rules={[{ required: true, message: "Please enter a display name" }]}
            >
              <Input placeholder="e.g., Customer Support Agent" />
            </Form.Item>

            <Form.Item
              label="Description"
              name="description"
              rules={[{ required: true, message: "Please enter a description" }]}
            >
              <Input.TextArea
                rows={3}
                placeholder="Describe what this agent does..."
              />
            </Form.Item>

            <Form.Item
              label="URL"
              name="url"
              rules={[
                { required: true, message: "Please enter the agent URL" },
                { type: "url", message: "Please enter a valid URL" }
              ]}
              tooltip="Base URL where the agent is hosted"
            >
              <Input placeholder="http://localhost:9999/" />
            </Form.Item>

            <Form.Item
              label="Version"
              name="version"
            >
              <Input placeholder="1.0.0" />
            </Form.Item>

            <Form.Item
              label="Protocol Version"
              name="protocolVersion"
            >
              <Input placeholder="1.0" />
            </Form.Item>
          </Panel>

          <Panel header="Skills (Required)" key="skills">
            <Form.List name="skills">
              {(fields, { add, remove }) => (
                <>
                  {fields.map((field, index) => (
                    <div key={field.key} style={{ marginBottom: 16, padding: 16, border: '1px solid #d9d9d9', borderRadius: 4 }}>
                      <Form.Item
                        {...field}
                        label="Skill ID"
                        name={[field.name, 'id']}
                        rules={[{ required: true, message: 'Required' }]}
                      >
                        <Input placeholder="e.g., hello_world" />
                      </Form.Item>
                      <Form.Item
                        {...field}
                        label="Skill Name"
                        name={[field.name, 'name']}
                        rules={[{ required: true, message: 'Required' }]}
                      >
                        <Input placeholder="e.g., Returns hello world" />
                      </Form.Item>
                      <Form.Item
                        {...field}
                        label="Description"
                        name={[field.name, 'description']}
                        rules={[{ required: true, message: 'Required' }]}
                      >
                        <Input.TextArea rows={2} placeholder="What this skill does" />
                      </Form.Item>
                      <Form.Item
                        {...field}
                        label="Tags (comma-separated)"
                        name={[field.name, 'tags']}
                        rules={[{ required: true, message: 'Required' }]}
                        getValueFromEvent={(e) => e.target.value.split(',').map((s: string) => s.trim())}
                        getValueProps={(value) => ({ value: Array.isArray(value) ? value.join(', ') : value })}
                      >
                        <Input placeholder="e.g., hello world, greeting" />
                      </Form.Item>
                      <Form.Item
                        {...field}
                        label="Examples (comma-separated)"
                        name={[field.name, 'examples']}
                        getValueFromEvent={(e) => e.target.value.split(',').map((s: string) => s.trim()).filter((s: string) => s)}
                        getValueProps={(value) => ({ value: Array.isArray(value) ? value.join(', ') : '' })}
                      >
                        <Input placeholder="e.g., hi, hello world" />
                      </Form.Item>
                      <AntButton 
                        type="link" 
                        danger 
                        onClick={() => remove(field.name)}
                        icon={<MinusCircleOutlined />}
                      >
                        Remove Skill
                      </AntButton>
                    </div>
                  ))}
                  <AntButton 
                    type="dashed" 
                    onClick={() => add()} 
                    icon={<PlusOutlined />}
                    style={{ width: '100%' }}
                  >
                    Add Skill
                  </AntButton>
                </>
              )}
            </Form.List>
          </Panel>

          <Panel header="Capabilities" key="capabilities">
            <Form.Item
              label="Streaming"
              name="streaming"
              valuePropName="checked"
            >
              <Switch />
            </Form.Item>

            <Form.Item
              label="Push Notifications"
              name="pushNotifications"
              valuePropName="checked"
            >
              <Switch />
            </Form.Item>

            <Form.Item
              label="State Transition History"
              name="stateTransitionHistory"
              valuePropName="checked"
            >
              <Switch />
            </Form.Item>
          </Panel>

          <Panel header="Optional Settings" key="optional">
            <Form.Item
              label="Icon URL"
              name="iconUrl"
            >
              <Input placeholder="https://example.com/icon.png" />
            </Form.Item>

            <Form.Item
              label="Documentation URL"
              name="documentationUrl"
            >
              <Input placeholder="https://docs.example.com" />
            </Form.Item>

            <Form.Item
              label="Supports Authenticated Extended Card"
              name="supportsAuthenticatedExtendedCard"
              valuePropName="checked"
            >
              <Switch />
            </Form.Item>
          </Panel>
        </Collapse>

        <Form.Item>
          <div style={{ display: "flex", justifyContent: "flex-end", gap: "8px" }}>
            <AntButton onClick={handleCancel}>
              Cancel
            </AntButton>
            <AntButton
              htmlType="submit"
              loading={isSubmitting}
            >
              Create Agent
            </AntButton>
          </div>
        </Form.Item>
      </Form>
    </Modal>
  );
};

export default AddAgentForm;

