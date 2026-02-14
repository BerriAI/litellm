import React, { useState } from "react";
import { Select, Typography, message } from "antd";
import { Button, TextInput } from "@tremor/react";
import { ArrowLeftIcon, PlusIcon } from "@heroicons/react/outline";
import { DotsVerticalIcon } from "@heroicons/react/solid";
import { GuardrailPipeline, PipelineStep, PolicyCreateRequest, PolicyUpdateRequest, Policy } from "./types";
import { Guardrail } from "../guardrails/types";
import NotificationsManager from "../molecules/notifications_manager";

const { Text } = Typography;

const ACTION_OPTIONS = [
  { label: "Next Step", value: "next" },
  { label: "Allow", value: "allow" },
  { label: "Block", value: "block" },
  { label: "Custom Response", value: "modify_response" },
];

const ACTION_LABELS: Record<string, string> = {
  allow: "Allow",
  block: "Block",
  next: "Next Step",
  modify_response: "Custom Response",
};

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
// Icons (matching the reference image)
// ─────────────────────────────────────────────────────────────────────────────

const GuardrailIcon: React.FC = () => (
  <div
    style={{
      width: 28,
      height: 28,
      borderRadius: "50%",
      backgroundColor: "#eef2ff",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      flexShrink: 0,
    }}
  >
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#6366f1" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <path d="M12 8v4" />
    </svg>
  </div>
);

const PlayIcon: React.FC = () => (
  <div
    style={{
      width: 28,
      height: 28,
      borderRadius: "50%",
      backgroundColor: "#f3f4f6",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      flexShrink: 0,
    }}
  >
    <svg width="12" height="12" viewBox="0 0 24 24" fill="#6b7280" stroke="none">
      <polygon points="6,3 20,12 6,21" />
    </svg>
  </div>
);

const PassIcon: React.FC = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
    <circle cx="12" cy="12" r="10" />
    <path d="M9 12l2 2 4-4" />
  </svg>
);

const FailIcon: React.FC = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#f87171" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
    <circle cx="12" cy="12" r="10" />
  </svg>
);

// ─────────────────────────────────────────────────────────────────────────────
// Connector
// ─────────────────────────────────────────────────────────────────────────────

interface ConnectorProps {
  onInsert: () => void;
}

const Connector: React.FC<ConnectorProps> = ({ onInsert }) => (
  <div className="flex flex-col items-center" style={{ height: 56 }}>
    <div style={{ width: 1, flex: 1, backgroundColor: "#d1d5db" }} />
    <button
      onClick={onInsert}
      className="flex items-center justify-center"
      style={{
        width: 24,
        height: 24,
        borderRadius: "50%",
        border: "1px solid #d1d5db",
        backgroundColor: "#fff",
        cursor: "pointer",
        zIndex: 1,
        transition: "all 0.15s ease",
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = "#6366f1";
        e.currentTarget.style.backgroundColor = "#eef2ff";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = "#d1d5db";
        e.currentTarget.style.backgroundColor = "#fff";
      }}
      title="Insert step"
    >
      <PlusIcon style={{ width: 12, height: 12, color: "#9ca3af" }} />
    </button>
    <div style={{ width: 1, flex: 1, backgroundColor: "#d1d5db" }} />
  </div>
);

// ─────────────────────────────────────────────────────────────────────────────
// Step Card (editable)
// ─────────────────────────────────────────────────────────────────────────────

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
        borderRadius: 10,
        padding: "16px 20px",
        backgroundColor: "#fff",
        maxWidth: 720,
        width: "100%",
      }}
    >
      {/* Header row: icon + GUARDRAIL label ... Step N + menu */}
      <div className="flex items-center justify-between" style={{ marginBottom: 12 }}>
        <div className="flex items-center gap-2">
          <GuardrailIcon />
          <span
            style={{
              fontSize: 11,
              fontWeight: 700,
              textTransform: "uppercase",
              color: "#6366f1",
              letterSpacing: "0.06em",
            }}
          >
            GUARDRAIL
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span style={{ fontSize: 13, color: "#9ca3af" }}>
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
            }}
            title="Delete step"
          >
            <DotsVerticalIcon style={{ width: 16, height: 16, color: "#9ca3af" }} />
          </button>
        </div>
      </div>

      {/* Guardrail name / selector */}
      <div style={{ marginBottom: 12 }}>
        <Select
          showSearch
          style={{ width: "100%" }}
          placeholder="Select a guardrail"
          value={step.guardrail || undefined}
          onChange={(value) => onChange({ guardrail: value })}
          options={guardrailOptions}
          filterOption={(input, option) =>
            (option?.label ?? "").toString().toLowerCase().includes(input.toLowerCase())
          }
        />
      </div>

      {/* Divider */}
      <div style={{ borderTop: "1px solid #f3f4f6", marginBottom: 12 }} />

      {/* Pass / Fail row */}
      <div className="flex items-center gap-6" style={{ flexWrap: "wrap" }}>
        <div className="flex items-center gap-2">
          <PassIcon />
          <span style={{ fontSize: 13, color: "#374151" }}>Pass</span>
          <span style={{ fontSize: 13, color: "#9ca3af" }}>&#8594;</span>
          <Select
            size="small"
            style={{ minWidth: 140 }}
            value={step.on_pass}
            onChange={(value) => onChange({ on_pass: value as PipelineStep["on_pass"] })}
            options={ACTION_OPTIONS}
          />
        </div>
        <div className="flex items-center gap-2">
          <FailIcon />
          <span style={{ fontSize: 13, color: "#374151" }}>Fail</span>
          <span style={{ fontSize: 13, color: "#9ca3af" }}>&#8594;</span>
          <Select
            size="small"
            style={{ minWidth: 140 }}
            value={step.on_fail}
            onChange={(value) => onChange({ on_fail: value as PipelineStep["on_fail"] })}
            options={ACTION_OPTIONS}
          />
        </div>
      </div>

      {/* Custom response input */}
      {(step.on_pass === "modify_response" || step.on_fail === "modify_response") && (
        <div style={{ marginTop: 10 }}>
          <TextInput
            placeholder="Custom response message"
            value={step.modify_response_message || ""}
            onChange={(e) => onChange({ modify_response_message: e.target.value || null })}
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
    <div className="flex flex-col items-center" style={{ padding: "16px 0" }}>
      {/* Trigger Card */}
      <div
        style={{
          border: "1px solid #e5e7eb",
          borderRadius: 10,
          padding: "16px 20px",
          backgroundColor: "#fff",
          maxWidth: 720,
          width: "100%",
        }}
      >
        <div className="flex items-center gap-3">
          <PlayIcon />
          <div>
            <span
              style={{
                fontSize: 11,
                fontWeight: 700,
                textTransform: "uppercase",
                color: "#6b7280",
                letterSpacing: "0.06em",
                display: "block",
                marginBottom: 2,
              }}
            >
              TRIGGER
            </span>
            <span style={{ fontSize: 14, fontWeight: 600, color: "#111827", display: "block" }}>
              Incoming LLM Request
            </span>
            <span style={{ fontSize: 13, color: "#9ca3af" }}>
              This flow runs when a request matches this policy
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

      {/* Bottom connector */}
      <Connector onInsert={() => handleInsertStep(pipeline.steps.length)} />
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Read-only display for policy info view
// ─────────────────────────────────────────────────────────────────────────────

interface PipelineInfoDisplayProps {
  pipeline: GuardrailPipeline;
}

export const PipelineInfoDisplay: React.FC<PipelineInfoDisplayProps> = ({ pipeline }) => (
  <div className="flex flex-col items-center" style={{ padding: "16px 0" }}>
    {/* Trigger */}
    <div
      style={{
        border: "1px solid #e5e7eb",
        borderRadius: 10,
        padding: "14px 20px",
        backgroundColor: "#fff",
        maxWidth: 720,
        width: "100%",
      }}
    >
      <div className="flex items-center gap-3">
        <PlayIcon />
        <div>
          <span style={{ fontSize: 11, fontWeight: 700, textTransform: "uppercase", color: "#6b7280", letterSpacing: "0.06em", display: "block", marginBottom: 2 }}>
            TRIGGER
          </span>
          <span style={{ fontSize: 14, fontWeight: 600, color: "#111827" }}>
            Incoming LLM Request
          </span>
        </div>
      </div>
    </div>

    {/* Steps */}
    {pipeline.steps.map((step, index) => (
      <React.Fragment key={index}>
        {/* Connector */}
        <div style={{ width: 1, height: 32, backgroundColor: "#d1d5db" }} />

        {/* Step card */}
        <div
          style={{
            border: "1px solid #e5e7eb",
            borderRadius: 10,
            padding: "14px 20px",
            backgroundColor: "#fff",
            maxWidth: 720,
            width: "100%",
          }}
        >
          {/* Header */}
          <div className="flex items-center justify-between" style={{ marginBottom: 8 }}>
            <div className="flex items-center gap-2">
              <GuardrailIcon />
              <span style={{ fontSize: 11, fontWeight: 700, textTransform: "uppercase", color: "#6366f1", letterSpacing: "0.06em" }}>
                GUARDRAIL
              </span>
            </div>
            <span style={{ fontSize: 13, color: "#9ca3af" }}>Step {index + 1}</span>
          </div>

          {/* Name */}
          <div style={{ fontSize: 15, fontWeight: 600, color: "#111827", marginBottom: 8 }}>
            {step.guardrail}
          </div>

          {/* Divider */}
          <div style={{ borderTop: "1px solid #f3f4f6", marginBottom: 10 }} />

          {/* Pass / Fail */}
          <div className="flex items-center gap-6" style={{ fontSize: 13, color: "#374151" }}>
            <span className="flex items-center gap-1.5">
              <PassIcon /> Pass &#8594; {ACTION_LABELS[step.on_pass] || step.on_pass}
            </span>
            <span className="flex items-center gap-1.5">
              <FailIcon /> Fail &#8594; {ACTION_LABELS[step.on_fail] || step.on_fail}
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
        backgroundColor: "#f9fafb",
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
        <div className="flex items-center gap-3">
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
            <ArrowLeftIcon style={{ width: 18, height: 18, color: "#6b7280" }} />
          </button>
          <span style={{ fontSize: 14, color: "#6b7280" }}>Policies</span>
          <span style={{ fontSize: 14, color: "#d1d5db" }}>/</span>
          <TextInput
            placeholder="Policy name..."
            value={policyName}
            onChange={(e) => setPolicyName(e.target.value)}
            disabled={isEditing}
            style={{ width: 240 }}
          />
          <span
            style={{
              fontSize: 11,
              fontWeight: 600,
              backgroundColor: "#eef2ff",
              color: "#6366f1",
              padding: "3px 8px",
              borderRadius: 4,
              letterSpacing: "0.02em",
            }}
          >
            Flow
          </span>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="secondary" onClick={onBack}>
            Cancel
          </Button>
          <Button onClick={handleSave} loading={isSubmitting}>
            {isEditing ? "Update Policy" : "Save Policy"}
          </Button>
        </div>
      </div>

      {/* Description bar */}
      <div
        style={{
          padding: "8px 24px",
          backgroundColor: "#fff",
          borderBottom: "1px solid #e5e7eb",
          flexShrink: 0,
        }}
      >
        <TextInput
          placeholder="Add a description (optional)..."
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          style={{ maxWidth: 500 }}
        />
      </div>

      {/* Flow builder canvas */}
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          display: "flex",
          justifyContent: "center",
          padding: "32px 24px",
        }}
      >
        <div style={{ maxWidth: 760, width: "100%" }}>
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
