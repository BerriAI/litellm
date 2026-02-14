import React, { useState } from "react";
import { Select, Typography, message } from "antd";
import { Button, TextInput } from "@tremor/react";
import { ArrowLeftIcon, PlusIcon, TrashIcon } from "@heroicons/react/outline";
import { GuardrailPipeline, PipelineStep, PolicyCreateRequest, PolicyUpdateRequest, Policy } from "./types";
import { Guardrail } from "../guardrails/types";
import NotificationsManager from "../molecules/notifications_manager";

const { Text } = Typography;

const ACTION_OPTIONS = [
  { label: "Continue to next step", value: "next" },
  { label: "Allow (end flow)", value: "allow" },
  { label: "Block with error response", value: "block" },
  { label: "Return custom response", value: "modify_response" },
];

function createDefaultStep(): PipelineStep {
  return {
    guardrail: "",
    on_pass: "next",
    on_fail: "block",
    pass_data: false,
    modify_response_message: null,
  };
}

function insertStep(steps: PipelineStep[], atIndex: number): PipelineStep[] {
  const newSteps = [...steps];
  newSteps.splice(atIndex, 0, createDefaultStep());
  return newSteps;
}

function removeStep(steps: PipelineStep[], index: number): PipelineStep[] {
  if (steps.length <= 1) return steps;
  const newSteps = [...steps];
  newSteps.splice(index, 1);
  return newSteps;
}

function updateStepAtIndex(
  steps: PipelineStep[],
  index: number,
  updated: Partial<PipelineStep>
): PipelineStep[] {
  return steps.map((s, i) => (i === index ? { ...s, ...updated } : s));
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────────────────────────────────────

interface ConnectorProps {
  onInsert: () => void;
}

const Connector: React.FC<ConnectorProps> = ({ onInsert }) => (
  <div className="flex flex-col items-center" style={{ height: 28 }}>
    <div style={{ width: 1, flex: 1, backgroundColor: "#e5e7eb" }} />
    <button
      onClick={onInsert}
      className="flex items-center justify-center"
      style={{
        width: 20,
        height: 20,
        borderRadius: "50%",
        border: "1px solid #e5e7eb",
        backgroundColor: "#fff",
        cursor: "pointer",
        zIndex: 1,
        marginTop: -2,
        marginBottom: -2,
        transition: "all 0.15s ease",
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = "#9ca3af";
        e.currentTarget.style.backgroundColor = "#f9fafb";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = "#e5e7eb";
        e.currentTarget.style.backgroundColor = "#fff";
      }}
      title="Insert step"
    >
      <PlusIcon style={{ width: 10, height: 10, color: "#9ca3af" }} />
    </button>
    <div style={{ width: 1, flex: 1, backgroundColor: "#e5e7eb" }} />
  </div>
);

interface StepCardProps {
  step: PipelineStep;
  stepIndex: number;
  totalSteps: number;
  onChange: (updated: Partial<PipelineStep>) => void;
  onDelete: () => void;
  availableGuardrails: Guardrail[];
}

const StepCard: React.FC<StepCardProps> = ({
  step,
  stepIndex,
  totalSteps,
  onChange,
  onDelete,
  availableGuardrails,
}) => {
  const guardrailOptions = availableGuardrails.map((g) => ({
    label: g.guardrail_name || g.guardrail_id,
    value: g.guardrail_name || g.guardrail_id,
  }));

  return (
    <div
      style={{
        border: "1px solid #e5e7eb",
        borderRadius: 6,
        padding: "10px 14px",
        backgroundColor: "#fff",
        maxWidth: 680,
        width: "100%",
      }}
    >
      {/* Row 1: GUARDRAIL label + selector + step number + delete */}
      <div className="flex items-center gap-2" style={{ marginBottom: 8 }}>
        <span
          style={{
            fontSize: 10,
            fontWeight: 600,
            textTransform: "uppercase",
            color: "#9ca3af",
            letterSpacing: "0.05em",
            flexShrink: 0,
          }}
        >
          GUARDRAIL
        </span>
        <Select
          showSearch
          size="small"
          style={{ flex: 1 }}
          placeholder="Select a guardrail"
          value={step.guardrail || undefined}
          onChange={(value) => onChange({ guardrail: value })}
          options={guardrailOptions}
          filterOption={(input, option) =>
            (option?.label ?? "").toString().toLowerCase().includes(input.toLowerCase())
          }
        />
        <span style={{ fontSize: 11, color: "#9ca3af", flexShrink: 0 }}>
          Step {stepIndex + 1}
        </span>
        <button
          onClick={onDelete}
          disabled={totalSteps <= 1}
          style={{
            background: "none",
            border: "none",
            cursor: totalSteps <= 1 ? "not-allowed" : "pointer",
            opacity: totalSteps <= 1 ? 0.3 : 1,
            padding: 2,
            display: "flex",
            alignItems: "center",
            transition: "all 0.15s ease",
          }}
          onMouseEnter={(e) => {
            if (totalSteps > 1) {
              const icon = e.currentTarget.querySelector("svg");
              if (icon) (icon as SVGElement).style.color = "#ef4444";
            }
          }}
          onMouseLeave={(e) => {
            const icon = e.currentTarget.querySelector("svg");
            if (icon) (icon as SVGElement).style.color = "#9ca3af";
          }}
          title="Delete step"
        >
          <TrashIcon style={{ width: 14, height: 14, color: "#9ca3af", transition: "color 0.15s ease" }} />
        </button>
      </div>

      {/* Row 2: ON PASS + ON FAIL inline */}
      <div className="flex items-center gap-4" style={{ flexWrap: "wrap" }}>
        <div className="flex items-center gap-2" style={{ flex: 1, minWidth: 200 }}>
          <span
            style={{
              fontSize: 10,
              fontWeight: 600,
              textTransform: "uppercase",
              color: "#16a34a",
              letterSpacing: "0.05em",
              flexShrink: 0,
            }}
          >
            ON PASS
          </span>
          <Select
            size="small"
            style={{ flex: 1 }}
            value={step.on_pass}
            onChange={(value) => onChange({ on_pass: value as PipelineStep["on_pass"] })}
            options={ACTION_OPTIONS}
          />
        </div>
        <div className="flex items-center gap-2" style={{ flex: 1, minWidth: 200 }}>
          <span
            style={{
              fontSize: 10,
              fontWeight: 600,
              textTransform: "uppercase",
              color: "#dc2626",
              letterSpacing: "0.05em",
              flexShrink: 0,
            }}
          >
            ON FAIL
          </span>
          <Select
            size="small"
            style={{ flex: 1 }}
            value={step.on_fail}
            onChange={(value) => onChange({ on_fail: value as PipelineStep["on_fail"] })}
            options={ACTION_OPTIONS}
          />
        </div>
      </div>

      {/* Custom response input (shown only when modify_response is selected) */}
      {(step.on_pass === "modify_response" || step.on_fail === "modify_response") && (
        <div style={{ marginTop: 6 }}>
          <TextInput
            placeholder="Custom response message"
            value={step.modify_response_message || ""}
            onChange={(e) => onChange({ modify_response_message: e.target.value || null })}
            style={{ fontSize: 13 }}
          />
        </div>
      )}
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Main Component
// ─────────────────────────────────────────────────────────────────────────────

interface PipelineFlowBuilderProps {
  pipeline: GuardrailPipeline;
  onChange: (pipeline: GuardrailPipeline) => void;
  availableGuardrails: Guardrail[];
}

const PipelineFlowBuilder: React.FC<PipelineFlowBuilderProps> = ({
  pipeline,
  onChange,
  availableGuardrails,
}) => {
  const handleInsertStep = (atIndex: number) => {
    onChange({ ...pipeline, steps: insertStep(pipeline.steps, atIndex) });
  };

  const handleRemoveStep = (index: number) => {
    onChange({ ...pipeline, steps: removeStep(pipeline.steps, index) });
  };

  const handleUpdateStep = (index: number, updated: Partial<PipelineStep>) => {
    onChange({
      ...pipeline,
      steps: updateStepAtIndex(pipeline.steps, index, updated),
    });
  };

  return (
    <div className="flex flex-col items-center" style={{ padding: "8px 0" }}>
      {/* Trigger Card */}
      <div
        style={{
          border: "1px solid #e5e7eb",
          borderRadius: 6,
          padding: "10px 14px",
          backgroundColor: "#fafafa",
          maxWidth: 680,
          width: "100%",
        }}
      >
        <div className="flex items-center gap-2">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="#9ca3af" stroke="none">
            <polygon points="5,3 19,12 5,21" />
          </svg>
          <div>
            <span
              style={{
                fontSize: 10,
                fontWeight: 600,
                textTransform: "uppercase",
                color: "#9ca3af",
                letterSpacing: "0.05em",
              }}
            >
              TRIGGER
            </span>
            <span style={{ fontSize: 13, fontWeight: 500, marginLeft: 8 }}>
              Incoming LLM Request
            </span>
          </div>
        </div>
      </div>

      {/* Steps */}
      {pipeline.steps.map((step, index) => (
        <React.Fragment key={index}>
          <Connector onInsert={() => handleInsertStep(index)} />
          <StepCard
            step={step}
            stepIndex={index}
            totalSteps={pipeline.steps.length}
            onChange={(updated) => handleUpdateStep(index, updated)}
            onDelete={() => handleRemoveStep(index)}
            availableGuardrails={availableGuardrails}
          />
        </React.Fragment>
      ))}

      {/* Bottom add button */}
      <Connector onInsert={() => handleInsertStep(pipeline.steps.length)} />
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Read-only display for policy info view
// ─────────────────────────────────────────────────────────────────────────────

const ACTION_LABELS: Record<string, string> = {
  allow: "Allow",
  block: "Block",
  next: "Next Step",
  modify_response: "Custom Response",
};

interface PipelineInfoDisplayProps {
  pipeline: GuardrailPipeline;
}

export const PipelineInfoDisplay: React.FC<PipelineInfoDisplayProps> = ({ pipeline }) => (
  <div className="flex flex-col items-center" style={{ padding: "8px 0" }}>
    {/* Trigger */}
    <div
      style={{
        border: "1px solid #e5e7eb",
        borderRadius: 6,
        padding: "8px 14px",
        backgroundColor: "#fafafa",
        maxWidth: 680,
        width: "100%",
      }}
    >
      <div className="flex items-center gap-2">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="#9ca3af" stroke="none">
          <polygon points="5,3 19,12 5,21" />
        </svg>
        <span style={{ fontSize: 10, fontWeight: 600, textTransform: "uppercase", color: "#9ca3af", letterSpacing: "0.05em" }}>
          TRIGGER
        </span>
        <span style={{ fontSize: 13, fontWeight: 500 }}>
          Incoming LLM Request
        </span>
      </div>
    </div>

    {/* Steps */}
    {pipeline.steps.map((step, index) => (
      <React.Fragment key={index}>
        <div style={{ width: 1, height: 20, backgroundColor: "#e5e7eb" }} />
        <div
          style={{
            border: "1px solid #e5e7eb",
            borderRadius: 6,
            padding: "8px 14px",
            backgroundColor: "#fff",
            maxWidth: 680,
            width: "100%",
          }}
        >
          <div className="flex items-center gap-2" style={{ marginBottom: 4 }}>
            <span style={{ fontSize: 10, fontWeight: 600, textTransform: "uppercase", color: "#9ca3af", letterSpacing: "0.05em" }}>
              GUARDRAIL
            </span>
            <span style={{ fontSize: 13, fontWeight: 500 }}>{step.guardrail}</span>
            <span style={{ fontSize: 11, color: "#9ca3af", marginLeft: "auto" }}>Step {index + 1}</span>
          </div>
          <div className="flex gap-4" style={{ fontSize: 12, color: "#6b7280" }}>
            <span>
              <span style={{ fontSize: 10, fontWeight: 600, color: "#16a34a", textTransform: "uppercase" }}>Pass</span>
              {" "}
              {ACTION_LABELS[step.on_pass] || step.on_pass}
            </span>
            <span>
              <span style={{ fontSize: 10, fontWeight: 600, color: "#dc2626", textTransform: "uppercase" }}>Fail</span>
              {" "}
              {ACTION_LABELS[step.on_fail] || step.on_fail}
            </span>
          </div>
        </div>
      </React.Fragment>
    ))}
  </div>
);

// ─────────────────────────────────────────────────────────────────────────────
// Full-screen Flow Builder Page
// ─────────────────────────────────────────────────────────────────────────────

interface FlowBuilderPageProps {
  onBack: () => void;
  onSuccess: () => void;
  accessToken: string | null;
  editingPolicy?: Policy | null;
  availableGuardrails: Guardrail[];
  createPolicy: (accessToken: string, policyData: any) => Promise<any>;
  updatePolicy: (accessToken: string, policyId: string, policyData: any) => Promise<any>;
}

export const FlowBuilderPage: React.FC<FlowBuilderPageProps> = ({
  onBack,
  onSuccess,
  accessToken,
  editingPolicy,
  availableGuardrails,
  createPolicy,
  updatePolicy,
}) => {
  const isEditing = !!editingPolicy?.policy_id;

  const [policyName, setPolicyName] = useState(editingPolicy?.policy_name || "");
  const [description, setDescription] = useState(editingPolicy?.description || "");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [pipeline, setPipeline] = useState<GuardrailPipeline>(
    editingPolicy?.pipeline || { mode: "pre_call", steps: [createDefaultStep()] }
  );

  const handleSave = async () => {
    if (!policyName.trim()) {
      message.error("Please enter a policy name");
      return;
    }
    if (!accessToken) {
      message.error("No access token available");
      return;
    }

    const emptySteps = pipeline.steps.filter((s) => !s.guardrail);
    if (emptySteps.length > 0) {
      message.error("Please select a guardrail for all steps");
      return;
    }

    setIsSubmitting(true);
    try {
      const guardrailsFromPipeline = pipeline.steps
        .map((s) => s.guardrail)
        .filter(Boolean);

      const data: PolicyCreateRequest | PolicyUpdateRequest = {
        policy_name: policyName,
        description: description || undefined,
        guardrails_add: guardrailsFromPipeline,
        guardrails_remove: [],
        pipeline: pipeline,
      };

      if (isEditing && editingPolicy) {
        await updatePolicy(accessToken, editingPolicy.policy_id, data as PolicyUpdateRequest);
        NotificationsManager.success("Policy updated successfully");
      } else {
        await createPolicy(accessToken, data as PolicyCreateRequest);
        NotificationsManager.success("Policy created successfully");
      }

      onSuccess();
      onBack();
    } catch (error) {
      console.error("Failed to save policy:", error);
      NotificationsManager.fromBackend(
        "Failed to save policy: " + (error instanceof Error ? error.message : String(error))
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: "#fafafa",
        zIndex: 1000,
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
      }}
    >
      {/* Header bar */}
      <div
        style={{
          borderBottom: "1px solid #e5e7eb",
          backgroundColor: "#fff",
          padding: "10px 24px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexShrink: 0,
        }}
      >
        <div className="flex items-center gap-2">
          <button
            onClick={onBack}
            style={{
              background: "none",
              border: "none",
              cursor: "pointer",
              padding: 4,
              display: "flex",
              alignItems: "center",
            }}
          >
            <ArrowLeftIcon style={{ width: 16, height: 16, color: "#9ca3af" }} />
          </button>
          <span style={{ fontSize: 13, color: "#9ca3af" }}>Policies</span>
          <span style={{ fontSize: 13, color: "#d1d5db" }}>/</span>
          <TextInput
            placeholder="Policy name..."
            value={policyName}
            onChange={(e) => setPolicyName(e.target.value)}
            disabled={isEditing}
            style={{ width: 220, fontSize: 13 }}
          />
          <span
            style={{
              fontSize: 10,
              fontWeight: 600,
              backgroundColor: "#f3f4f6",
              color: "#6b7280",
              padding: "2px 6px",
              borderRadius: 3,
              textTransform: "uppercase",
              letterSpacing: "0.05em",
            }}
          >
            Flow
          </span>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="secondary" onClick={onBack} style={{ fontSize: 13 }}>
            Cancel
          </Button>
          <Button
            onClick={handleSave}
            loading={isSubmitting}
            style={{ fontSize: 13 }}
          >
            {isEditing ? "Update Policy" : "Save Policy"}
          </Button>
        </div>
      </div>

      {/* Description bar */}
      <div
        style={{
          padding: "6px 24px",
          backgroundColor: "#fff",
          borderBottom: "1px solid #e5e7eb",
          flexShrink: 0,
        }}
      >
        <TextInput
          placeholder="Add a description (optional)..."
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          style={{ maxWidth: 480, fontSize: 13 }}
        />
      </div>

      {/* Flow builder canvas */}
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          display: "flex",
          justifyContent: "center",
          padding: "24px 24px",
        }}
      >
        <div style={{ maxWidth: 720, width: "100%" }}>
          <PipelineFlowBuilder
            pipeline={pipeline}
            onChange={setPipeline}
            availableGuardrails={availableGuardrails}
          />
        </div>
      </div>
    </div>
  );
};

export { createDefaultStep };
export default PipelineFlowBuilder;
