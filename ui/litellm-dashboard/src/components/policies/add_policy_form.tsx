import React, { useState, useEffect } from "react";
import { Form, Select, Modal, Divider, Typography, Tag, Alert, Radio } from "antd";
import { Button, TextInput, Textarea } from "@tremor/react";
import { Policy, PolicyCreateRequest, PolicyUpdateRequest } from "./types";
import { Guardrail } from "../guardrails/types";
import { getResolvedGuardrails, modelAvailableCall } from "../networking";
import NotificationsManager from "../molecules/notifications_manager";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

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
  createPolicy: (accessToken: string, policyData: any) => Promise<any>;
  updatePolicy: (accessToken: string, policyId: string, policyData: any) => Promise<any>;
}

const AddPolicyForm: React.FC<AddPolicyFormProps> = ({
  visible,
  onClose,
  onSuccess,
  accessToken,
  editingPolicy,
  existingPolicies,
  availableGuardrails,
  createPolicy,
  updatePolicy,
}) => {
  const [form] = Form.useForm();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [resolvedGuardrails, setResolvedGuardrails] = useState<string[]>([]);
  const [isLoadingResolved, setIsLoadingResolved] = useState(false);
  const [modelConditionType, setModelConditionType] = useState<"model" | "regex">("model");
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const { userId, userRole } = useAuthorized();

  const isEditing = !!editingPolicy;

  useEffect(() => {
    if (visible && editingPolicy) {
      const modelCondition = editingPolicy.condition?.model;
      // Detect if it's a regex pattern (contains *, ., [, ], etc.)
      const isRegex = modelCondition && /[.*+?^${}()|[\]\\]/.test(modelCondition);
      setModelConditionType(isRegex ? "regex" : "model");

      form.setFieldsValue({
        policy_name: editingPolicy.policy_name,
        description: editingPolicy.description,
        inherit: editingPolicy.inherit,
        guardrails_add: editingPolicy.guardrails_add || [],
        guardrails_remove: editingPolicy.guardrails_remove || [],
        model_condition: modelCondition,
      });
      // Load resolved guardrails for editing
      if (editingPolicy.policy_id && accessToken) {
        loadResolvedGuardrails(editingPolicy.policy_id);
      }
    } else if (visible) {
      form.resetFields();
      setResolvedGuardrails([]);
      setModelConditionType("model");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visible, editingPolicy, form]);

  useEffect(() => {
    if (visible && accessToken) {
      loadAvailableModels();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visible, accessToken]);

  const loadAvailableModels = async () => {
    if (!accessToken) return;

    try {
      const response = await modelAvailableCall(accessToken, userId, userRole);
      if (response?.data) {
        const models = response.data.map((m: any) => m.id || m.model_name).filter(Boolean);
        setAvailableModels(models);
      }
    } catch (error) {
      console.error("Failed to load available models:", error);
    }
  };

  const loadResolvedGuardrails = async (policyId: string) => {
    if (!accessToken) return;

    setIsLoadingResolved(true);
    try {
      const data = await getResolvedGuardrails(accessToken, policyId);
      setResolvedGuardrails(data.resolved_guardrails || []);
    } catch (error) {
      console.error("Failed to load resolved guardrails:", error);
    } finally {
      setIsLoadingResolved(false);
    }
  };

  const computeResolvedGuardrails = (): string[] => {
    const values = form.getFieldsValue(true);
    const inheritFrom = values.inherit;
    const guardrailsAdd = values.guardrails_add || [];
    const guardrailsRemove = values.guardrails_remove || [];

    let resolved = new Set<string>();

    // If inheriting, find parent policy and get its guardrails
    if (inheritFrom) {
      const parentPolicy = existingPolicies.find(p => p.policy_name === inheritFrom);
      if (parentPolicy) {
        // Recursively resolve parent's guardrails
        const parentResolved = resolveParentGuardrails(parentPolicy);
        parentResolved.forEach(g => resolved.add(g));
      }
    }

    // Add guardrails
    guardrailsAdd.forEach((g: string) => resolved.add(g));

    // Remove guardrails
    guardrailsRemove.forEach((g: string) => resolved.delete(g));

    return Array.from(resolved).sort();
  };

  const resolveParentGuardrails = (policy: Policy): string[] => {
    let resolved = new Set<string>();

    // If parent inherits, resolve recursively
    if (policy.inherit) {
      const grandparent = existingPolicies.find(p => p.policy_name === policy.inherit);
      if (grandparent) {
        const grandparentResolved = resolveParentGuardrails(grandparent);
        grandparentResolved.forEach(g => resolved.add(g));
      }
    }

    // Add parent's guardrails
    if (policy.guardrails_add) {
      policy.guardrails_add.forEach(g => resolved.add(g));
    }

    // Remove parent's removed guardrails
    if (policy.guardrails_remove) {
      policy.guardrails_remove.forEach(g => resolved.delete(g));
    }

    return Array.from(resolved);
  };

  // Recompute resolved guardrails when form values change
  const handleFormChange = () => {
    const resolved = computeResolvedGuardrails();
    setResolvedGuardrails(resolved);
  };

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
        await updatePolicy(accessToken, editingPolicy.policy_id, data as PolicyUpdateRequest);
        NotificationsManager.success("Policy updated successfully");
      } else {
        await createPolicy(accessToken, data as PolicyCreateRequest);
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
        onValuesChange={handleFormChange}
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

        {resolvedGuardrails.length > 0 && (
          <Alert
            message="Resolved Guardrails"
            description={
              <div>
                <Text type="secondary" style={{ display: "block", marginBottom: 8 }}>
                  These are the final guardrails that will be applied (including inheritance):
                </Text>
                <div className="flex flex-wrap gap-1">
                  {resolvedGuardrails.map((g) => (
                    <Tag key={g} color="blue">
                      {g}
                    </Tag>
                  ))}
                </div>
              </div>
            }
            type="info"
            showIcon
            style={{ marginBottom: 16 }}
          />
        )}

        <Divider orientation="left">
          <Text strong>Conditions (Optional)</Text>
        </Divider>

        <Alert
          message="Model Scope"
          description="By default, this policy will run on all models. You can optionally restrict it to specific models below."
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
        />

        <Form.Item label="Model Condition Type">
          <Radio.Group
            value={modelConditionType}
            onChange={(e) => {
              setModelConditionType(e.target.value);
              form.setFieldValue("model_condition", undefined);
            }}
          >
            <Radio value="model">Select Model</Radio>
            <Radio value="regex">Custom Regex Pattern</Radio>
          </Radio.Group>
        </Form.Item>

        <Form.Item
          name="model_condition"
          label={modelConditionType === "model" ? "Model (Optional)" : "Regex Pattern (Optional)"}
          tooltip={
            modelConditionType === "model"
              ? "Select a specific model to apply this policy to. Leave empty to apply to all models."
              : "Enter a regex pattern to match models (e.g., gpt-4.* or bedrock/.*). Leave empty to apply to all models."
          }
        >
          {modelConditionType === "model" ? (
            <Select
              showSearch
              allowClear
              placeholder="Leave empty to apply to all models"
              options={availableModels.map((model) => ({
                label: model,
                value: model,
              }))}
              filterOption={(input, option) =>
                (option?.label ?? "").toLowerCase().includes(input.toLowerCase())
              }
              style={{ width: "100%" }}
            />
          ) : (
            <TextInput placeholder="Leave empty to apply to all models (e.g., gpt-4.* or bedrock/claude-.*)" />
          )}
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
