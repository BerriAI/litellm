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
  onOpenFlowBuilder: () => void;
  accessToken: string | null;
  editingPolicy?: Policy | null;
  existingPolicies: Policy[];
  availableGuardrails: Guardrail[];
  createPolicy: (accessToken: string, policyData: any) => Promise<any>;
  updatePolicy: (accessToken: string, policyId: string, policyData: any) => Promise<any>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Mode Picker (Step 1) - shown first when creating a new policy
// ─────────────────────────────────────────────────────────────────────────────

interface ModePicker {
  selected: "simple" | "flow_builder";
  onSelect: (mode: "simple" | "flow_builder") => void;
}

const ModePicker: React.FC<ModePicker> = ({ selected, onSelect }) => (
  <div className="flex gap-4" style={{ padding: "8px 0" }}>
    {/* Simple Mode Card */}
    <div
      onClick={() => onSelect("simple")}
      style={{
        flex: 1,
        padding: "24px 20px",
        border: `2px solid ${selected === "simple" ? "#4f46e5" : "#e5e7eb"}`,
        borderRadius: 12,
        cursor: "pointer",
        backgroundColor: selected === "simple" ? "#eef2ff" : "#fff",
        transition: "all 0.15s ease",
      }}
    >
      <div
        style={{
          width: 40,
          height: 40,
          borderRadius: 10,
          backgroundColor: selected === "simple" ? "#e0e7ff" : "#f3f4f6",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          marginBottom: 16,
        }}
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={selected === "simple" ? "#4f46e5" : "#6b7280"} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <rect x="3" y="3" width="18" height="18" rx="2" />
          <path d="M8 7h8M8 12h8M8 17h5" />
        </svg>
      </div>
      <Text strong style={{ fontSize: 15, display: "block", marginBottom: 4 }}>
        Simple Mode
      </Text>
      <Text type="secondary" style={{ fontSize: 13 }}>
        Pick guardrails from a list. All run in parallel.
      </Text>
    </div>

    {/* Flow Builder Card */}
    <div
      onClick={() => onSelect("flow_builder")}
      style={{
        flex: 1,
        padding: "24px 20px",
        border: `2px solid ${selected === "flow_builder" ? "#4f46e5" : "#e5e7eb"}`,
        borderRadius: 12,
        cursor: "pointer",
        backgroundColor: selected === "flow_builder" ? "#eef2ff" : "#fff",
        transition: "all 0.15s ease",
        position: "relative",
      }}
    >
      <Tag
        color="purple"
        style={{
          position: "absolute",
          top: 12,
          right: 12,
          fontSize: 10,
          fontWeight: 600,
          margin: 0,
        }}
      >
        NEW
      </Tag>
      <div
        style={{
          width: 40,
          height: 40,
          borderRadius: 10,
          backgroundColor: selected === "flow_builder" ? "#e0e7ff" : "#f3f4f6",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          marginBottom: 16,
        }}
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={selected === "flow_builder" ? "#4f46e5" : "#6b7280"} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
        </svg>
      </div>
      <Text strong style={{ fontSize: 15, display: "block", marginBottom: 4 }}>
        Flow Builder
      </Text>
      <Text type="secondary" style={{ fontSize: 13 }}>
        Define steps, conditions, and error responses.
      </Text>
    </div>
  </div>
);

// ─────────────────────────────────────────────────────────────────────────────
// Main Component
// ─────────────────────────────────────────────────────────────────────────────

const AddPolicyForm: React.FC<AddPolicyFormProps> = ({
  visible,
  onClose,
  onSuccess,
  onOpenFlowBuilder,
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
  const [step, setStep] = useState<"pick_mode" | "simple_form">("pick_mode");
  const [selectedMode, setSelectedMode] = useState<"simple" | "flow_builder">("simple");
  const { userId, userRole } = useAuthorized();

  // Only consider it "editing" if editingPolicy has a policy_id (real existing policy)
  const isEditing = !!editingPolicy?.policy_id;

  useEffect(() => {
    if (visible && editingPolicy) {
      const modelCondition = editingPolicy.condition?.model;
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

      if (editingPolicy.policy_id && accessToken) {
        loadResolvedGuardrails(editingPolicy.policy_id);
      }

      // If editing a pipeline policy, go directly to flow builder
      if (editingPolicy.pipeline) {
        onClose();
        onOpenFlowBuilder();
        return;
      }
      // If editing a simple policy, skip mode picker
      setStep("simple_form");
    } else if (visible) {
      form.resetFields();
      setResolvedGuardrails([]);
      setModelConditionType("model");
      setSelectedMode("simple");
      setStep("pick_mode");
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

    if (inheritFrom) {
      const parentPolicy = existingPolicies.find(p => p.policy_name === inheritFrom);
      if (parentPolicy) {
        const parentResolved = resolveParentGuardrails(parentPolicy);
        parentResolved.forEach(g => resolved.add(g));
      }
    }

    guardrailsAdd.forEach((g: string) => resolved.add(g));
    guardrailsRemove.forEach((g: string) => resolved.delete(g));

    return Array.from(resolved).sort();
  };

  const resolveParentGuardrails = (policy: Policy): string[] => {
    let resolved = new Set<string>();

    if (policy.inherit) {
      const grandparent = existingPolicies.find(p => p.policy_name === policy.inherit);
      if (grandparent) {
        resolveParentGuardrails(grandparent).forEach(g => resolved.add(g));
      }
    }
    if (policy.guardrails_add) {
      policy.guardrails_add.forEach(g => resolved.add(g));
    }
    if (policy.guardrails_remove) {
      policy.guardrails_remove.forEach(g => resolved.delete(g));
    }
    return Array.from(resolved);
  };

  const handleFormChange = () => {
    setResolvedGuardrails(computeResolvedGuardrails());
  };

  const resetForm = () => {
    form.resetFields();
  };

  const handleClose = () => {
    resetForm();
    setStep("pick_mode");
    setSelectedMode("simple");
    onClose();
  };

  const handleModeConfirm = () => {
    if (selectedMode === "flow_builder") {
      onClose();
      onOpenFlowBuilder();
    } else {
      setStep("simple_form");
    }
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

  // ── Mode Picker Step ──────────────────────────────────────────────────────
  if (step === "pick_mode") {
    return (
      <Modal
        title="Create New Policy"
        open={visible}
        onCancel={handleClose}
        footer={null}
        width={620}
      >
        <ModePicker selected={selectedMode} onSelect={setSelectedMode} />

        {selectedMode === "flow_builder" && (
          <Alert
            message="You'll be redirected to the full-screen Flow Builder to design your policy logic visually."
            type="info"
            style={{
              marginTop: 16,
              backgroundColor: "#eef2ff",
              border: "1px solid #c7d2fe",
            }}
          />
        )}

        <div className="flex justify-end gap-2" style={{ marginTop: 24 }}>
          <Button variant="secondary" onClick={handleClose}>
            Cancel
          </Button>
          <Button
            onClick={handleModeConfirm}
            style={{
              backgroundColor: "#4f46e5",
              color: "#fff",
              border: "none",
            }}
          >
            {selectedMode === "flow_builder" ? "Continue to Builder" : "Create Policy"}
          </Button>
        </div>
      </Modal>
    );
  }

  // ── Simple Form Step ──────────────────────────────────────────────────────
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
