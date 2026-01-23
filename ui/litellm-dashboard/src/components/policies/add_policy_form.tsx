import React, { useState, useEffect } from "react";
import { Form, Select, Modal, Divider, Typography } from "antd";
import { Button, TextInput, Textarea } from "@tremor/react";
import { Policy, PolicyCreateRequest, PolicyUpdateRequest } from "./types";
import { Guardrail } from "../guardrails/types";
import { createPolicyCall, updatePolicyCall } from "../networking";
import NotificationsManager from "../molecules/notifications_manager";

const { Text } = Typography;
const { Option } = Select;

interface AddPolicyFormProps {
  visible: boolean;
  onClose: () => void;
  onSuccess: () => void;
  accessToken: string | null;
  editingPolicy?: Policy | null;
  existingPolicies: Policy[];
  availableGuardrails: Guardrail[];
}

const AddPolicyForm: React.FC<AddPolicyFormProps> = ({
  visible,
  onClose,
  onSuccess,
  accessToken,
  editingPolicy,
  existingPolicies,
  availableGuardrails,
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

  const resetForm = () => {
    form.resetFields();
  };

  const handleClose = () => {
    resetForm();
    onClose();
  };

  const handleSubmit = async () => {
    try {
      setIsSubmitting(true);
      await form.validateFields();
      const values = form.getFieldsValue(true);

      if (!accessToken) {
        throw new Error("No access token available");
      }

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
        await updatePolicyCall(accessToken, editingPolicy.policy_id, data as PolicyUpdateRequest);
        NotificationsManager.success("Policy updated successfully");
      } else {
        await createPolicyCall(accessToken, data as PolicyCreateRequest);
        NotificationsManager.success("Policy created successfully");
      }

      resetForm();
      onSuccess();
      onClose();
    } catch (error) {
      console.error("Failed to save policy:", error);
      NotificationsManager.fromBackend(
        "Failed to save policy: " + (error instanceof Error ? error.message : String(error))
      );
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
      onCancel={handleClose}
      footer={null}
      width={700}
    >
      <Form
        form={form}
        layout="vertical"
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
          <TextInput
            placeholder="e.g., global-baseline, healthcare-compliance"
            disabled={isEditing}
          />
        </Form.Item>

        <Form.Item name="description" label="Description">
          <Textarea
            rows={2}
            placeholder="Describe what this policy does..."
          />
        </Form.Item>

        <Divider orientation="left">
          <Text strong>Inheritance</Text>
        </Divider>

        <Form.Item
          name="inherit"
          label="Inherit From"
          tooltip="Inherit guardrails from another policy. The child policy will include all guardrails from the parent."
        >
          <Select
            allowClear
            placeholder="Select a parent policy (optional)"
            options={policyOptions}
            style={{ width: "100%" }}
          />
        </Form.Item>

        <Divider orientation="left">
          <Text strong>Guardrails</Text>
        </Divider>

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
            style={{ width: "100%" }}
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
            style={{ width: "100%" }}
          />
        </Form.Item>

        <Divider orientation="left">
          <Text strong>Conditions (Optional)</Text>
        </Divider>

        <Form.Item
          name="model_condition"
          label="Model Condition"
          tooltip="Only apply this policy when the model matches this pattern (supports regex)"
        >
          <TextInput placeholder="e.g., gpt-4.* or bedrock/claude-3" />
        </Form.Item>

        <div className="flex justify-end space-x-2 mt-4">
          <Button variant="secondary" onClick={handleClose}>
            Cancel
          </Button>
          <Button onClick={handleSubmit} loading={isSubmitting}>
            {isEditing ? "Update Policy" : "Create Policy"}
          </Button>
        </div>
      </Form>
    </Modal>
  );
};

export default AddPolicyForm;
