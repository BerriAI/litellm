import React, { useState } from "react";
import {
  Modal,
  Form,
  Input,
  Select,
  Button,
  Space,
  Radio,
  Divider,
} from "antd";
import { Policy, PolicyAttachmentCreateRequest } from "./types";

interface AddAttachmentFormProps {
  visible: boolean;
  onClose: () => void;
  onSuccess: () => void;
  accessToken: string | null;
  policies: Policy[];
  onCreateAttachment: (data: PolicyAttachmentCreateRequest) => Promise<void>;
}

const AddAttachmentForm: React.FC<AddAttachmentFormProps> = ({
  visible,
  onClose,
  onSuccess,
  accessToken,
  policies,
  onCreateAttachment,
}) => {
  const [form] = Form.useForm();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [scopeType, setScopeType] = useState<"global" | "specific">("global");

  const handleSubmit = async (values: any) => {
    if (!accessToken) return;

    setIsSubmitting(true);
    try {
      const data: PolicyAttachmentCreateRequest = {
        policy_name: values.policy_name,
      };

      if (scopeType === "global") {
        data.scope = "*";
      } else {
        if (values.teams && values.teams.length > 0) {
          data.teams = values.teams;
        }
        if (values.keys && values.keys.length > 0) {
          data.keys = values.keys;
        }
        if (values.models && values.models.length > 0) {
          data.models = values.models;
        }
      }

      await onCreateAttachment(data);
      onSuccess();
      onClose();
      form.resetFields();
      setScopeType("global");
    } catch (error) {
      console.error("Error creating attachment:", error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const policyOptions = policies.map((p) => ({
    label: p.policy_name,
    value: p.policy_name,
  }));

  return (
    <Modal
      title="Create Policy Attachment"
      open={visible}
      onCancel={onClose}
      footer={null}
      width={600}
    >
      <Form
        form={form}
        layout="vertical"
        onFinish={handleSubmit}
        initialValues={{
          scope_type: "global",
        }}
      >
        <Form.Item
          name="policy_name"
          label="Policy"
          rules={[{ required: true, message: "Please select a policy" }]}
        >
          <Select
            placeholder="Select a policy to attach"
            options={policyOptions}
            showSearch
            filterOption={(input, option) =>
              (option?.label ?? "").toLowerCase().includes(input.toLowerCase())
            }
          />
        </Form.Item>

        <Divider orientation="left">Scope</Divider>

        <Form.Item label="Scope Type">
          <Radio.Group
            value={scopeType}
            onChange={(e) => setScopeType(e.target.value)}
          >
            <Radio value="global">Global (applies to all requests)</Radio>
            <Radio value="specific">Specific (teams, keys, or models)</Radio>
          </Radio.Group>
        </Form.Item>

        {scopeType === "specific" && (
          <>
            <Form.Item
              name="teams"
              label="Teams"
              tooltip="Team aliases this attachment applies to. Supports wildcards (e.g., healthcare-*)"
            >
              <Select
                mode="tags"
                placeholder="Enter team aliases (e.g., healthcare-team)"
                tokenSeparators={[","]}
              />
            </Form.Item>

            <Form.Item
              name="keys"
              label="Keys"
              tooltip="Key aliases this attachment applies to. Supports wildcards (e.g., dev-*)"
            >
              <Select
                mode="tags"
                placeholder="Enter key aliases (e.g., dev-key-*)"
                tokenSeparators={[","]}
              />
            </Form.Item>

            <Form.Item
              name="models"
              label="Models"
              tooltip="Model names this attachment applies to. Supports wildcards (e.g., gpt-4*)"
            >
              <Select
                mode="tags"
                placeholder="Enter model names (e.g., gpt-4, bedrock/*)"
                tokenSeparators={[","]}
              />
            </Form.Item>
          </>
        )}

        <Form.Item>
          <Space>
            <Button onClick={onClose}>Cancel</Button>
            <Button type="primary" htmlType="submit" loading={isSubmitting}>
              Create Attachment
            </Button>
          </Space>
        </Form.Item>
      </Form>
    </Modal>
  );
};

export default AddAttachmentForm;
