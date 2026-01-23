import React, { useState, useEffect } from "react";
import {
  Modal,
  Form,
  Input,
  Select,
  Button,
  Space,
  Typography,
  Divider,
} from "antd";
import { PlusOutlined, MinusCircleOutlined } from "@ant-design/icons";
import { Policy, PolicyCreateRequest, PolicyUpdateRequest } from "./types";
import { Guardrail } from "../guardrails/types";

const { TextArea } = Input;
const { Text } = Typography;

interface AddPolicyFormProps {
  visible: boolean;
  onClose: () => void;
  onSuccess: () => void;
  accessToken: string | null;
  editingPolicy?: Policy | null;
  existingPolicies: Policy[];
  availableGuardrails: Guardrail[];
  onCreatePolicy: (data: PolicyCreateRequest) => Promise<void>;
  onUpdatePolicy: (policyId: string, data: PolicyUpdateRequest) => Promise<void>;
}

const AddPolicyForm: React.FC<AddPolicyFormProps> = ({
  visible,
  onClose,
  onSuccess,
  accessToken,
  editingPolicy,
  existingPolicies,
  availableGuardrails,
  onCreatePolicy,
  onUpdatePolicy,
}) => {
  const [form] = Form.useForm();
  const [isSubmitting, setIsSubmitting] = useState(false);

  const isEditing = !!editingPolicy;

  useEffect(() => {
    if (visible && editingPolicy) {
      form.setFieldsValue({
        policy_name: editingPolicy.policy_name,
        description: editingPolicy.description,
        inherit: editingPolicy.inherit,
        guardrails_add: editingPolicy.guardrails_add || [],
        guardrails_remove: editingPolicy.guardrails_remove || [],
        model_condition: editingPolicy.condition?.model,
      });
    } else if (visible) {
      form.resetFields();
    }
  }, [visible, editingPolicy, form]);

  const handleSubmit = async (values: any) => {
    if (!accessToken) return;

    setIsSubmitting(true);
    try {
      const data: PolicyCreateRequest | PolicyUpdateRequest = {
        policy_name: values.policy_name,
        description: values.description || undefined,
        inherit: values.inherit || undefined,
        guardrails_add: values.guardrails_add || [],
        guardrails_remove: values.guardrails_remove || [],
        condition: values.model_condition
          ? { model: values.model_condition }
          : undefined,
      };

      if (isEditing && editingPolicy) {
        await onUpdatePolicy(editingPolicy.policy_id, data as PolicyUpdateRequest);
      } else {
        await onCreatePolicy(data as PolicyCreateRequest);
      }

      onSuccess();
      onClose();
      form.resetFields();
    } catch (error) {
      console.error("Error saving policy:", error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const guardrailOptions = availableGuardrails.map((g) => ({
    label: g.guardrail_name || g.guardrail_id,
    value: g.guardrail_name || g.guardrail_id,
  }));

  const policyOptions = existingPolicies
    .filter((p) => !editingPolicy || p.policy_id !== editingPolicy.policy_id)
    .map((p) => ({
      label: p.policy_name,
      value: p.policy_name,
    }));

  return (
    <Modal
      title={isEditing ? "Edit Policy" : "Create New Policy"}
      open={visible}
      onCancel={onClose}
      footer={null}
      width={700}
    >
      <Form
        form={form}
        layout="vertical"
        onFinish={handleSubmit}
        initialValues={{
          guardrails_add: [],
          guardrails_remove: [],
        }}
      >
        <Form.Item
          name="policy_name"
          label="Policy Name"
          rules={[
            { required: true, message: "Please enter a policy name" },
            {
              pattern: /^[a-zA-Z0-9_-]+$/,
              message:
                "Policy name can only contain letters, numbers, hyphens, and underscores",
            },
          ]}
        >
          <Input
            placeholder="e.g., global-baseline, healthcare-compliance"
            disabled={isEditing}
          />
        </Form.Item>

        <Form.Item name="description" label="Description">
          <TextArea
            rows={2}
            placeholder="Describe what this policy does..."
          />
        </Form.Item>

        <Divider orientation="left">Inheritance</Divider>

        <Form.Item
          name="inherit"
          label="Inherit From"
          tooltip="Inherit guardrails from another policy. The child policy will include all guardrails from the parent."
        >
          <Select
            allowClear
            placeholder="Select a parent policy (optional)"
            options={policyOptions}
          />
        </Form.Item>

        <Divider orientation="left">Guardrails</Divider>

        <Form.Item
          name="guardrails_add"
          label="Guardrails to Add"
          tooltip="These guardrails will be added to requests matching this policy"
        >
          <Select
            mode="multiple"
            allowClear
            placeholder="Select guardrails to add"
            options={guardrailOptions}
          />
        </Form.Item>

        <Form.Item
          name="guardrails_remove"
          label="Guardrails to Remove"
          tooltip="These guardrails will be removed from inherited guardrails"
        >
          <Select
            mode="multiple"
            allowClear
            placeholder="Select guardrails to remove (from inherited)"
            options={guardrailOptions}
          />
        </Form.Item>

        <Divider orientation="left">Conditions (Optional)</Divider>

        <Form.Item
          name="model_condition"
          label="Model Condition"
          tooltip="Only apply this policy when the model matches this pattern (supports regex)"
        >
          <Input placeholder="e.g., gpt-4.* or bedrock/claude-3" />
        </Form.Item>

        <Form.Item>
          <Space>
            <Button onClick={onClose}>Cancel</Button>
            <Button type="primary" htmlType="submit" loading={isSubmitting}>
              {isEditing ? "Update Policy" : "Create Policy"}
            </Button>
          </Space>
        </Form.Item>
      </Form>
    </Modal>
  );
};

export default AddPolicyForm;
