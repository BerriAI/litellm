import React, { useState, useEffect, useMemo } from "react";
import { Select, Typography, message, Modal, Spin, Divider, Form } from "antd";
import { Button, TextInput } from "@tremor/react";
import {
  ArrowLeftIcon,
  PlusIcon,
  ChevronUpIcon,
  ChevronDownIcon,
  ClockIcon,
  CheckCircleIcon,
  SwitchHorizontalIcon,
} from "@heroicons/react/outline";
import { DotsVerticalIcon, PencilIcon, BeakerIcon } from "@heroicons/react/solid";
import {
  GuardrailPipeline,
  PipelineStep,
  PipelineTestResult,
  PolicyCreateRequest,
  PolicyUpdateRequest,
  Policy,
  PolicyVersionListResponse,
} from "./types";
import { Guardrail } from "../guardrails/types";
import { testPipelineCall, listPolicyVersions, createPolicyVersion, updatePolicyVersionStatus } from "../networking";
import NotificationsManager from "../molecules/notifications_manager";
import GuardrailInfoView from "../guardrails/guardrail_info";
import VersionStatusBadge from "./version_status_badge";
import VersionComparison from "./version_comparison";
import { getFrameworks } from "@/data/compliancePrompts";

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

/** Fake policy with multiple versions for demo (no backend). Name: "Flow builder demo". */
const DEMO_POLICY_NAME = "Flow builder demo";

function createDemoVersion(
  policyId: string,
  versionNumber: number,
  versionStatus: "draft" | "published" | "production",
  steps: PipelineStep[]
): Policy {
  return {
    policy_id: policyId,
    policy_name: DEMO_POLICY_NAME,
    inherit: null,
    description: `Demo version ${versionNumber} – switch, edit, and test without saving.`,
    guardrails_add: steps.map((s) => s.guardrail).filter(Boolean),
    guardrails_remove: [],
    condition: null,
    pipeline: { mode: "pre_call", steps },
    version_number: versionNumber,
    version_status: versionStatus,
    parent_version_id: null,
    is_latest: versionNumber === 2,
  };
}

const MOCK_DEMO_VERSIONS: Policy[] = [
  createDemoVersion("demo-v1", 1, "production", [
    { ...createDefaultStep(), guardrail: "pii_masking", on_pass: "next", on_fail: "block" },
  ]),
  createDemoVersion("demo-v2", 2, "draft", [
    { ...createDefaultStep(), guardrail: "pii_masking", on_pass: "next", on_fail: "block" },
    { ...createDefaultStep(), guardrail: "prompt_injection", on_pass: "allow", on_fail: "block" },
  ]),
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

/** Build initial pipeline from a policy (uses pipeline if present, else guardrails_add as steps). */
function getInitialPipelineFromPolicy(policy: Policy | null | undefined): GuardrailPipeline {
  if (!policy) return { mode: "pre_call", steps: [createDefaultStep()] };
  if (
    policy.pipeline?.steps &&
    Array.isArray(policy.pipeline.steps) &&
    policy.pipeline.steps.length > 0
  ) {
    return policy.pipeline;
  }
  const add = policy.guardrails_add || [];
  if (add.length === 0) return { mode: "pre_call", steps: [createDefaultStep()] };
  return {
    mode: "pre_call",
    steps: add.map((guardrail) => ({ ...createDefaultStep(), guardrail })),
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
  onEditGuardrail: (guardrailName: string) => void;
  onTestGuardrail: (guardrailName: string) => void;
}

const StepCard: React.FC<StepCardProps> = ({
  step,
  stepIndex,
  totalSteps,
  onChange,
  onDelete,
  availableGuardrails,
  onEditGuardrail,
  onTestGuardrail,
}) => {
  const guardrailOptions = availableGuardrails.map((g) => ({
    label: g.guardrail_name || g.guardrail_id,
    value: g.guardrail_name || g.guardrail_id,
  }));

  const selectedGuardrail = step.guardrail;

  return (
    <div
      style={{
        border: "1px solid #e5e7eb",
        borderRadius: 10,
        backgroundColor: "#fff",
        maxWidth: 720,
        width: "100%",
        overflow: "hidden",
      }}
    >
      {/* Header row */}
      <div
        className="flex items-center justify-between"
        style={{ padding: "14px 20px 0 20px" }}
      >
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

      {/* Guardrail selector with action buttons */}
      <div style={{ padding: "12px 20px 16px 20px" }}>
        <label style={{ fontSize: 12, fontWeight: 500, color: "#6b7280", display: "block", marginBottom: 6 }}>
          Guardrail
        </label>
        <div className="flex items-center gap-2">
          <Select
            showSearch
            style={{ flex: 1 }}
            placeholder="Select a guardrail"
            value={step.guardrail || undefined}
            onChange={(value) => onChange({ guardrail: value })}
            options={guardrailOptions}
            filterOption={(input, option) =>
              (option?.label ?? "").toString().toLowerCase().includes(input.toLowerCase())
            }
          />
          {selectedGuardrail && (
            <>
              <button
                onClick={() => onEditGuardrail(selectedGuardrail)}
                style={{
                  padding: "6px 12px",
                  border: "1px solid #d1d5db",
                  borderRadius: 6,
                  backgroundColor: "#fff",
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  gap: 4,
                  fontSize: 13,
                  color: "#374151",
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
                title="Edit guardrail settings"
              >
                <PencilIcon style={{ width: 14, height: 14 }} />
                Edit
              </button>
              <button
                onClick={() => onTestGuardrail(selectedGuardrail)}
                style={{
                  padding: "6px 12px",
                  border: "1px solid #d1d5db",
                  borderRadius: 6,
                  backgroundColor: "#fff",
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  gap: 4,
                  fontSize: 13,
                  color: "#374151",
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
                title="Test this guardrail"
              >
                <BeakerIcon style={{ width: 14, height: 14 }} />
                Test
              </button>
            </>
          )}
        </div>
      </div>

      {/* ON PASS section */}
      <div style={{ borderTop: "1px solid #f0f0f0", padding: "14px 20px" }}>
        <div className="flex items-center gap-2" style={{ marginBottom: 8 }}>
          <PassIcon />
          <span style={{ fontSize: 13, fontWeight: 600, color: "#374151" }}>ON PASS</span>
        </div>
        <label style={{ fontSize: 12, fontWeight: 500, color: "#6b7280", display: "block", marginBottom: 6 }}>
          Action
        </label>
        <Select
          style={{ width: "100%" }}
          value={step.on_pass}
          onChange={(value) => onChange({ on_pass: value as PipelineStep["on_pass"] })}
          options={ACTION_OPTIONS}
        />
        {step.on_pass === "modify_response" && (
          <div style={{ marginTop: 8 }}>
            <label style={{ fontSize: 12, fontWeight: 500, color: "#6b7280", display: "block", marginBottom: 6 }}>
              Custom Response Message
            </label>
            <TextInput
              placeholder="Enter custom response..."
              value={step.modify_response_message || ""}
              onChange={(e) => onChange({ modify_response_message: e.target.value || null })}
            />
          </div>
        )}
      </div>

      {/* ON FAIL section */}
      <div style={{ borderTop: "1px solid #f0f0f0", padding: "14px 20px" }}>
        <div className="flex items-center gap-2" style={{ marginBottom: 8 }}>
          <FailIcon />
          <span style={{ fontSize: 13, fontWeight: 600, color: "#374151" }}>ON FAIL</span>
        </div>
        <label style={{ fontSize: 12, fontWeight: 500, color: "#6b7280", display: "block", marginBottom: 6 }}>
          Action
        </label>
        <Select
          style={{ width: "100%" }}
          value={step.on_fail}
          onChange={(value) => onChange({ on_fail: value as PipelineStep["on_fail"] })}
          options={ACTION_OPTIONS}
        />
        {step.on_fail === "modify_response" && (
          <div style={{ marginTop: 8 }}>
            <label style={{ fontSize: 12, fontWeight: 500, color: "#6b7280", display: "block", marginBottom: 6 }}>
              Custom Response Message
            </label>
            <TextInput
              placeholder="Enter custom response..."
              value={step.modify_response_message || ""}
              onChange={(e) => onChange({ modify_response_message: e.target.value || null })}
            />
          </div>
        )}
      </div>
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
  accessToken: string | null;
  isAdmin: boolean;
  onGuardrailUpdated?: () => void;
}

const PipelineFlowBuilder: React.FC<PipelineFlowBuilderProps> = ({
  pipeline,
  onChange,
  availableGuardrails,
  accessToken,
  isAdmin,
  onGuardrailUpdated,
}) => {
  const [editingGuardrailName, setEditingGuardrailName] = useState<string | null>(null);
  const [testingGuardrailName, setTestingGuardrailName] = useState<string | null>(null);

  const steps = Array.isArray(pipeline?.steps) ? pipeline.steps : [createDefaultStep()];
  const guardrails = availableGuardrails ?? [];
  const safePipeline: GuardrailPipeline = {
    mode: pipeline?.mode ?? "pre_call",
    steps,
  };

  const handleInsertStep = (atIndex: number) => {
    onChange({ ...safePipeline, steps: insertStep(steps, atIndex) });
  };

  const handleRemoveStep = (index: number) => {
    onChange({ ...safePipeline, steps: removeStep(steps, index) });
  };

  const handleUpdateStep = (index: number, updated: Partial<PipelineStep>) => {
    onChange({
      ...safePipeline,
      steps: updateStepAtIndex(steps, index, updated),
    });
  };

  const handleEditGuardrail = (guardrailName: string) => {
    setEditingGuardrailName(guardrailName);
  };

  const handleTestGuardrail = (guardrailName: string) => {
    setTestingGuardrailName(guardrailName);
  };

  const handleCloseEditModal = () => {
    setEditingGuardrailName(null);
    if (onGuardrailUpdated) {
      onGuardrailUpdated();
    }
  };

  const handleCloseTestModal = () => {
    setTestingGuardrailName(null);
  };

  // Find the guardrail ID for the selected guardrail name
  const getGuardrailId = (guardrailName: string | null): string | null => {
    if (!guardrailName) return null;
    const guardrail = guardrails.find(
      (g) => g.guardrail_name === guardrailName || g.guardrail_id === guardrailName
    );
    return guardrail?.guardrail_id || null;
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
      {steps.map((step, index) => (
        <React.Fragment key={index}>
          <Connector onInsert={() => handleInsertStep(index)} />
          <StepCard
            step={step}
            stepIndex={index}
            totalSteps={steps.length}
            onChange={(updated) => handleUpdateStep(index, updated)}
            onDelete={() => handleRemoveStep(index)}
            availableGuardrails={guardrails}
            onEditGuardrail={handleEditGuardrail}
            onTestGuardrail={handleTestGuardrail}
          />
        </React.Fragment>
      ))}

      {/* Bottom connector */}
      <Connector onInsert={() => handleInsertStep(steps.length)} />

      {/* End card */}
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
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#6b7280" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="3" width="18" height="18" rx="2" />
              <line x1="8" y1="12" x2="16" y2="12" />
            </svg>
          </div>
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
              END
            </span>
            <span style={{ fontSize: 14, fontWeight: 600, color: "#111827", display: "block" }}>
              Continue to LLM
            </span>
            <span style={{ fontSize: 13, color: "#9ca3af" }}>
              Request proceeds to the model
            </span>
          </div>
        </div>
      </div>

      {/* Edit Guardrail Modal */}
      <Modal
        title="Edit Guardrail"
        open={!!editingGuardrailName}
        onCancel={handleCloseEditModal}
        footer={null}
        width={900}
        destroyOnHidden
        zIndex={1200}
        styles={{ body: { padding: 0, maxHeight: "80vh", overflowY: "auto" } }}
      >
        {editingGuardrailName && getGuardrailId(editingGuardrailName) && (
          <GuardrailInfoView
            guardrailId={getGuardrailId(editingGuardrailName)!}
            onClose={handleCloseEditModal}
            accessToken={accessToken}
            isAdmin={isAdmin}
          />
        )}
      </Modal>

      {/* Test Guardrail Modal */}
      <Modal
        title="Test Guardrail"
        open={!!testingGuardrailName}
        onCancel={handleCloseTestModal}
        footer={null}
        width={1200}
        destroyOnHidden
        zIndex={1200}
        styles={{ body: { padding: "16px", maxHeight: "80vh", overflowY: "auto" } }}
      >
        {testingGuardrailName && (
          <div>
            <Text style={{ fontSize: 14, color: "#6b7280", marginBottom: 16, display: "block" }}>
              Testing guardrail: <strong>{testingGuardrailName}</strong>
            </Text>
            <PipelineTestPanel
              pipeline={{
                mode: pipeline.mode,
                steps: [
                  {
                    guardrail: testingGuardrailName,
                    on_pass: "next",
                    on_fail: "block",
                    pass_data: false,
                    modify_response_message: null,
                  },
                ],
              }}
              accessToken={accessToken}
              onClose={handleCloseTestModal}
            />
          </div>
        )}
      </Modal>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Read-only display for policy info view
// ─────────────────────────────────────────────────────────────────────────────

interface PipelineInfoDisplayProps {
  pipeline: GuardrailPipeline;
}

export const PipelineInfoDisplay: React.FC<PipelineInfoDisplayProps> = ({ pipeline }) => {
  const steps = Array.isArray(pipeline?.steps) ? pipeline.steps : [];
  return (
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
    {steps.map((step, index) => (
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
};

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline Test Panel (right drawer)
// ─────────────────────────────────────────────────────────────────────────────

interface PipelineTestPanelProps {
  pipeline: GuardrailPipeline;
  accessToken: string | null;
  onClose: () => void;
}

const OUTCOME_STYLES: Record<string, { bg: string; color: string; label: string }> = {
  pass: { bg: "#f0fdf4", color: "#16a34a", label: "PASS" },
  fail: { bg: "#fef2f2", color: "#dc2626", label: "FAIL" },
  error: { bg: "#fffbeb", color: "#d97706", label: "ERROR" },
};

const TERMINAL_STYLES: Record<string, { bg: string; color: string }> = {
  allow: { bg: "#f0fdf4", color: "#16a34a" },
  block: { bg: "#fef2f2", color: "#dc2626" },
  modify_response: { bg: "#eff6ff", color: "#2563eb" },
};

type ComplianceTestResult = {
  promptId: string;
  prompt: string;
  expectedResult: "fail" | "pass";
  actualResult: "blocked" | "allowed";
  isMatch: boolean;
  status: "complete" | "pending" | "error";
  error?: string;
};

const PipelineTestPanel: React.FC<PipelineTestPanelProps> = ({
  pipeline,
  accessToken,
  onClose,
}) => {
  const [testMessage, setTestMessage] = useState("Hello, can you help me?");
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<PipelineTestResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const steps = Array.isArray(pipeline?.steps) ? pipeline.steps : [];

  const [testTab, setTestTab] = useState<"quick" | "compliance">("quick");
  const complianceFrameworks = useMemo(() => getFrameworks(), []);
  const allCompliancePrompts = useMemo(
    () => complianceFrameworks.flatMap((fw) => fw.categories.flatMap((c) => c.prompts)),
    [complianceFrameworks]
  );
  const [selectedPromptIds, setSelectedPromptIds] = useState<Set<string>>(new Set());
  const [complianceResults, setComplianceResults] = useState<ComplianceTestResult[]>([]);
  const [isRunningCompliance, setIsRunningCompliance] = useState(false);

  const togglePromptSelection = (id: string) => {
    setSelectedPromptIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const selectAllPrompts = () => {
    setSelectedPromptIds(new Set(allCompliancePrompts.map((p) => p.id)));
  };

  const handleRunTest = async () => {
    if (!accessToken) return;

    const emptySteps = steps.filter((s) => !s.guardrail);
    if (emptySteps.length > 0) {
      setError("All steps must have a guardrail selected");
      return;
    }

    setIsRunning(true);
    setResult(null);
    setError(null);

    try {
      const data = await testPipelineCall(
        accessToken,
        pipeline,
        [{ role: "user", content: testMessage }]
      );
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setIsRunning(false);
    }
  };

  const handleRunComplianceTests = async () => {
    if (!accessToken || selectedPromptIds.size === 0) return;

    const emptySteps = steps.filter((s) => !s.guardrail);
    if (emptySteps.length > 0) {
      setError("All steps must have a guardrail selected");
      return;
    }

    const selected = allCompliancePrompts.filter((p) => selectedPromptIds.has(p.id));
    setComplianceResults(
      selected.map((p) => ({
        promptId: p.id,
        prompt: p.prompt,
        expectedResult: p.expectedResult,
        actualResult: "allowed" as const,
        isMatch: false,
        status: "pending" as const,
      }))
    );
    setIsRunningCompliance(true);
    setError(null);

    for (let i = 0; i < selected.length; i++) {
      const p = selected[i];
      try {
        const data = await testPipelineCall(accessToken, pipeline, [
          { role: "user", content: p.prompt },
        ]);
        const actualResult: "blocked" | "allowed" =
          data.terminal_action === "block" ? "blocked" : "allowed";
        const isMatch =
          (p.expectedResult === "fail" && actualResult === "blocked") ||
          (p.expectedResult === "pass" && actualResult === "allowed");
        setComplianceResults((prev) =>
          prev.map((r) =>
            r.promptId === p.id
              ? { ...r, actualResult, isMatch, status: "complete" as const }
              : r
          )
        );
      } catch (e) {
        const errMsg = e instanceof Error ? e.message : String(e);
        setComplianceResults((prev) =>
          prev.map((r) =>
            r.promptId === p.id
              ? { ...r, status: "error" as const, error: errMsg, actualResult: "blocked" as const }
              : r
          )
        );
      }
    }
    setIsRunningCompliance(false);
  };

  return (
    <div
      style={{
        width: 440,
        borderLeft: "1px solid #e5e7eb",
        backgroundColor: "#fff",
        display: "flex",
        flexDirection: "column",
        flexShrink: 0,
        overflow: "hidden",
      }}
    >
      {/* Panel header */}
      <div
        style={{
          padding: "12px 16px",
          borderBottom: "1px solid #e5e7eb",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <span style={{ fontSize: 14, fontWeight: 600, color: "#111827" }}>Test Pipeline</span>
        <button
          onClick={onClose}
          style={{
            background: "none",
            border: "none",
            cursor: "pointer",
            fontSize: 18,
            color: "#9ca3af",
            padding: "0 4px",
          }}
        >
          x
        </button>
      </div>

      {/* Tabs: Quick test | Compliance */}
      <div
        style={{
          display: "flex",
          borderBottom: "1px solid #e5e7eb",
          padding: "0 16px",
        }}
      >
        <button
          type="button"
          onClick={() => setTestTab("quick")}
          style={{
            padding: "10px 12px",
            fontSize: 13,
            fontWeight: 500,
            border: "none",
            background: "none",
            cursor: "pointer",
            color: testTab === "quick" ? "#6366f1" : "#6b7280",
            borderBottom: testTab === "quick" ? "2px solid #6366f1" : "2px solid transparent",
            marginBottom: -1,
          }}
        >
          Quick test
        </button>
        <button
          type="button"
          onClick={() => setTestTab("compliance")}
          style={{
            padding: "10px 12px",
            fontSize: 13,
            fontWeight: 500,
            border: "none",
            background: "none",
            cursor: "pointer",
            color: testTab === "compliance" ? "#6366f1" : "#6b7280",
            borderBottom: testTab === "compliance" ? "2px solid #6366f1" : "2px solid transparent",
            marginBottom: -1,
          }}
        >
          Compliance datasets
        </button>
      </div>

      {testTab === "quick" && (
      <div style={{ padding: 16, borderBottom: "1px solid #e5e7eb" }}>
        <label style={{ fontSize: 12, fontWeight: 500, color: "#6b7280", display: "block", marginBottom: 6 }}>
          Test Message
        </label>
        <textarea
          value={testMessage}
          onChange={(e) => setTestMessage(e.target.value)}
          placeholder="Enter a test message..."
          rows={3}
          style={{
            width: "100%",
            border: "1px solid #d1d5db",
            borderRadius: 6,
            padding: "8px 10px",
            fontSize: 13,
            resize: "vertical",
            fontFamily: "inherit",
          }}
        />
        <Button
          onClick={handleRunTest}
          loading={isRunning}
          style={{ marginTop: 8, width: "100%" }}
        >
          Run Test
        </Button>
      </div>
      )}

      {testTab === "compliance" && (
      <div style={{ padding: 16, borderBottom: "1px solid #e5e7eb", overflowY: "auto", maxHeight: 280 }}>
        <p style={{ fontSize: 12, color: "#6b7280", marginBottom: 8 }}>
          Same datasets as Compliance playground. Select prompts and run.
        </p>
        <Button
          size="xs"
          variant="secondary"
          onClick={selectAllPrompts}
          style={{ marginBottom: 12 }}
        >
          Select all
        </Button>
        <div style={{ maxHeight: 160, overflowY: "auto", marginBottom: 12 }}>
          {complianceFrameworks.map((fw) => (
            <div key={fw.name} style={{ marginBottom: 8 }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: "#374151", marginBottom: 4 }}>
                {fw.name}
              </div>
              {fw.categories.map((cat) => (
                <div key={cat.name} style={{ marginLeft: 8, marginBottom: 4 }}>
                  <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 2 }}>{cat.name}</div>
                  {cat.prompts.map((p) => (
                    <label
                      key={p.id}
                      style={{
                        display: "flex",
                        alignItems: "flex-start",
                        gap: 6,
                        cursor: "pointer",
                        fontSize: 12,
                        marginBottom: 2,
                      }}
                    >
                      <input
                        type="checkbox"
                        checked={selectedPromptIds.has(p.id)}
                        onChange={() => togglePromptSelection(p.id)}
                        style={{ marginTop: 2 }}
                      />
                      <span className="line-clamp-2" style={{ color: "#111827" }}>
                        {p.prompt.slice(0, 80)}
                        {p.prompt.length > 80 ? "…" : ""}
                      </span>
                    </label>
                  ))}
                </div>
              ))}
            </div>
          ))}
        </div>
        <Button
          onClick={handleRunComplianceTests}
          loading={isRunningCompliance}
          disabled={selectedPromptIds.size === 0}
          style={{ width: "100%" }}
        >
          Run selected ({selectedPromptIds.size})
        </Button>
      </div>
      )}

      {/* Results section */}
      <div style={{ flex: 1, overflowY: "auto", padding: 16, minHeight: 0 }}>
        {error && (
          <div
            style={{
              padding: "10px 12px",
              backgroundColor: "#fef2f2",
              border: "1px solid #fecaca",
              borderRadius: 6,
              fontSize: 13,
              color: "#dc2626",
              marginBottom: 12,
            }}
          >
            {error}
          </div>
        )}

        {testTab === "compliance" && complianceResults.length > 0 && (
          <div style={{ marginBottom: 16 }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: "#111827", marginBottom: 8 }}>
              Results
            </div>
            <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 8 }}>
              {complianceResults.filter((r) => r.status === "complete").length} / {complianceResults.length} complete
              {" · "}
              {complianceResults.filter((r) => r.isMatch).length} match expected
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {complianceResults.map((r) => (
                <div
                  key={r.promptId}
                  style={{
                    border: "1px solid #e5e7eb",
                    borderRadius: 6,
                    padding: "8px 10px",
                    backgroundColor: r.status === "error" ? "#fef2f2" : r.isMatch ? "#f0fdf4" : "#fffbeb",
                  }}
                >
                  <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 4 }}>
                    Expected: {r.expectedResult} → Actual: {r.actualResult}
                    {r.status === "complete" && (
                      <span style={{ marginLeft: 6, fontWeight: 600 }}>
                        {r.isMatch ? "✓ Match" : "✗ Mismatch"}
                      </span>
                    )}
                    {r.status === "error" && r.error && (
                      <span style={{ color: "#dc2626", marginLeft: 4 }}>{r.error}</span>
                    )}
                  </div>
                  <div style={{ fontSize: 12, color: "#111827" }} className="line-clamp-2">
                    {r.prompt.slice(0, 120)}
                    {r.prompt.length > 120 ? "…" : ""}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {testTab === "quick" && result && (
          <div>
            {/* Step results */}
            {result.step_results.map((step, i) => {
              const style = OUTCOME_STYLES[step.outcome] || OUTCOME_STYLES.error;
              return (
                <div
                  key={i}
                  style={{
                    border: "1px solid #e5e7eb",
                    borderRadius: 8,
                    padding: "10px 12px",
                    marginBottom: 8,
                  }}
                >
                  <div className="flex items-center justify-between" style={{ marginBottom: 4 }}>
                    <span style={{ fontSize: 13, fontWeight: 600, color: "#111827" }}>
                      Step {i + 1}: {step.guardrail_name}
                    </span>
                    <span
                      style={{
                        fontSize: 11,
                        fontWeight: 700,
                        backgroundColor: style.bg,
                        color: style.color,
                        padding: "2px 8px",
                        borderRadius: 4,
                      }}
                    >
                      {style.label}
                    </span>
                  </div>
                  <div style={{ fontSize: 12, color: "#6b7280" }}>
                    Action: {ACTION_LABELS[step.action_taken] || step.action_taken}
                    {step.duration_seconds != null && (
                      <span style={{ marginLeft: 8 }}>
                        ({(step.duration_seconds * 1000).toFixed(0)}ms)
                      </span>
                    )}
                  </div>
                  {step.error_detail && (
                    <div style={{ fontSize: 12, color: "#dc2626", marginTop: 4 }}>
                      {step.error_detail}
                    </div>
                  )}
                </div>
              );
            })}

            {/* Terminal result */}
            <div
              style={{
                borderTop: "1px solid #e5e7eb",
                paddingTop: 12,
                marginTop: 4,
              }}
            >
              <div className="flex items-center justify-between">
                <span style={{ fontSize: 13, fontWeight: 600, color: "#111827" }}>Result</span>
                {(() => {
                  const ts = TERMINAL_STYLES[result.terminal_action] || TERMINAL_STYLES.block;
                  return (
                    <span
                      style={{
                        fontSize: 12,
                        fontWeight: 700,
                        backgroundColor: ts.bg,
                        color: ts.color,
                        padding: "3px 10px",
                        borderRadius: 4,
                        textTransform: "uppercase",
                      }}
                    >
                      {result.terminal_action === "modify_response" ? "Custom Response" : result.terminal_action}
                    </span>
                  );
                })()}
              </div>
              {result.error_message && (
                <div style={{ fontSize: 12, color: "#dc2626", marginTop: 6 }}>
                  {result.error_message}
                </div>
              )}
              {result.modify_response_message && (
                <div style={{ fontSize: 12, color: "#2563eb", marginTop: 6 }}>
                  Response: {result.modify_response_message}
                </div>
              )}
            </div>
          </div>
        )}

        {testTab === "quick" && !result && !error && (
          <div style={{ textAlign: "center", color: "#9ca3af", fontSize: 13, marginTop: 24 }}>
            Enter a test message and click "Run Test" to execute the pipeline
          </div>
        )}
        {testTab === "compliance" && complianceResults.length === 0 && !error && (
          <div style={{ textAlign: "center", color: "#9ca3af", fontSize: 13, marginTop: 24 }}>
            Select prompts above and click "Run selected" to test with compliance datasets
          </div>
        )}
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Version sidebar for Flow Builder (self-contained, used only in this page)
// ─────────────────────────────────────────────────────────────────────────────

interface FlowBuilderVersionSidebarProps {
  editingPolicy: Policy;
  accessToken: string | null;
  onVersionSelect: (version: Policy) => void;
  onVersionCreated: () => void;
  /** When set, use this list instead of fetching (demo mode). New Version only calls onVersionCreated. */
  versionsOverride?: Policy[] | null;
}

const FlowBuilderVersionSidebar: React.FC<FlowBuilderVersionSidebarProps> = ({
  editingPolicy,
  accessToken,
  onVersionSelect,
  onVersionCreated,
  versionsOverride,
}) => {
  const policyName = editingPolicy.policy_name;
  const currentPolicyId = editingPolicy.policy_id;
  const isDemoMode = versionsOverride != null;

  const [versions, setVersions] = useState<Policy[]>([]);
  const [isLoading, setIsLoading] = useState(!isDemoMode);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [compareModalOpen, setCompareModalOpen] = useState(false);
  const [comparePolicyId1, setComparePolicyId1] = useState<string>("");
  const [comparePolicyId2, setComparePolicyId2] = useState<string>("");

  const displayVersions = isDemoMode ? versionsOverride! : versions;

  const loadVersions = async () => {
    if (!accessToken || !policyName || isDemoMode) return;
    setIsLoading(true);
    try {
      const response: PolicyVersionListResponse = await listPolicyVersions(accessToken, policyName);
      setVersions(response.policies || []);
    } catch (error) {
      console.error("Failed to load versions:", error);
      const errMsg = error instanceof Error ? error.message : String(error);
      if (!errMsg.includes("column") && !errMsg.includes("version_number")) {
        NotificationsManager.fromBackend("Failed to load policy versions: " + errMsg);
      }
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (isDemoMode) {
      setIsLoading(false);
      return;
    }
    loadVersions();
  }, [policyName, accessToken, isDemoMode]);

  const handleCreateNewVersion = () => {
    if (isDemoMode) {
      Modal.confirm({
        title: "Create New Version",
        content: "Add a new draft version from the current pipeline? (Demo – not saved to server.)",
        okText: "Create",
        cancelText: "Cancel",
        onOk: () => {
          setActionLoading("create");
          onVersionCreated();
          setActionLoading(null);
        },
      });
      return;
    }
    if (!accessToken) return;
    Modal.confirm({
      title: "Create New Version",
      content: "Create a new draft version from the current policy?",
      okText: "Create",
      cancelText: "Cancel",
      onOk: async () => {
        setActionLoading("create");
        try {
          const newPolicy = await createPolicyVersion(accessToken, currentPolicyId);
          NotificationsManager.success("New version created successfully");
          await loadVersions();
          onVersionCreated();
          if (newPolicy?.policy_id) {
            onVersionSelect(newPolicy as Policy);
          }
        } catch (error) {
          console.error("Failed to create version:", error);
          const errMsg = error instanceof Error ? error.message : String(error);
          if (
            errMsg.includes("column") ||
            errMsg.includes("version_number") ||
            errMsg.includes("schema")
          ) {
            NotificationsManager.error(
              "Database migration required. Run: poetry run prisma migrate dev --name add_policy_versioning"
            );
          } else {
            NotificationsManager.fromBackend("Failed to create version: " + errMsg);
          }
        } finally {
          setActionLoading(null);
        }
      },
    });
  };

  const handlePromoteToPublished = async (policyId: string, versionNumber: number) => {
    if (!accessToken) return;
    Modal.confirm({
      title: "Promote to Published",
      content: `Promote version ${versionNumber} to Published status?`,
      okText: "Promote",
      cancelText: "Cancel",
      onOk: async () => {
        setActionLoading(policyId);
        try {
          await updatePolicyVersionStatus(accessToken, policyId, "published");
          NotificationsManager.success("Version promoted to Published");
          await loadVersions();
        } catch (error) {
          console.error("Failed to promote version:", error);
          NotificationsManager.fromBackend(
            "Failed to promote version: " + (error instanceof Error ? error.message : String(error))
          );
        } finally {
          setActionLoading(null);
        }
      },
    });
  };

  const handlePromoteToProduction = async (policyId: string, versionNumber: number) => {
    if (!accessToken) return;
    Modal.confirm({
      title: "Promote to Production",
      content: `Promote version ${versionNumber} to Production? Any existing production version will be demoted to Published.`,
      okText: "Promote",
      cancelText: "Cancel",
      okType: "primary",
      onOk: async () => {
        setActionLoading(policyId);
        try {
          await updatePolicyVersionStatus(accessToken, policyId, "production");
          NotificationsManager.success("Version promoted to Production");
          await loadVersions();
        } catch (error) {
          console.error("Failed to promote version:", error);
          NotificationsManager.fromBackend(
            "Failed to promote version: " + (error instanceof Error ? error.message : String(error))
          );
        } finally {
          setActionLoading(null);
        }
      },
    });
  };

  const handleDemote = async (policyId: string, versionNumber: number) => {
    if (!accessToken) return;
    Modal.confirm({
      title: "Demote Version",
      content: `Demote version ${versionNumber} from Production to Published?`,
      okText: "Demote",
      cancelText: "Cancel",
      onOk: async () => {
        setActionLoading(policyId);
        try {
          await updatePolicyVersionStatus(accessToken, policyId, "published");
          NotificationsManager.success("Version demoted to Published");
          await loadVersions();
        } catch (error) {
          console.error("Failed to demote version:", error);
          NotificationsManager.fromBackend(
            "Failed to demote version: " + (error instanceof Error ? error.message : String(error))
          );
        } finally {
          setActionLoading(null);
        }
      },
    });
  };

  const formatDate = (dateString?: string | null) => {
    if (!dateString) return "N/A";
    return new Date(dateString).toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  };

  const getActionButtons = (version: Policy) => {
    if (isDemoMode) return null;
    const isProcessing = actionLoading === version.policy_id;
    const status = version.version_status;

    if (status === "draft") {
      return (
        <Button
          size="xs"
          variant="secondary"
          icon={ChevronUpIcon}
          onClick={() => handlePromoteToPublished(version.policy_id, version.version_number ?? 1)}
          loading={!!isProcessing}
          disabled={!!actionLoading}
        >
          Publish
        </Button>
      );
    }
    if (status === "published") {
      return (
        <Button
          size="xs"
          variant="primary"
          icon={ChevronUpIcon}
          onClick={() =>
            handlePromoteToProduction(version.policy_id, version.version_number ?? 1)
          }
          loading={!!isProcessing}
          disabled={!!actionLoading}
        >
          To Production
        </Button>
      );
    }
    if (status === "production") {
      return (
        <Button
          size="xs"
          variant="secondary"
          icon={ChevronDownIcon}
          onClick={() => handleDemote(version.policy_id, version.version_number ?? 1)}
          loading={!!isProcessing}
          disabled={!!actionLoading}
        >
          Demote
        </Button>
      );
    }
    return null;
  };

  if (isLoading) {
    return (
      <div className="flex justify-center items-center p-8">
        <Spin size="default" />
      </div>
    );
  }

  return (
    <div className="space-y-4" style={{ padding: 12 }}>
      <div className="flex justify-between items-center mb-2" style={{ minHeight: 32 }}>
        <span className="text-sm font-medium m-0" style={{ lineHeight: "32px" }}>
          Versions
        </span>
        <Button
          size="xs"
          icon={PlusIcon}
          onClick={handleCreateNewVersion}
          loading={actionLoading === "create"}
          disabled={!!actionLoading}
        >
          New Version
        </Button>
      </div>
      <Divider style={{ margin: "12px 0" }} />
      <div className="space-y-3">
        {displayVersions.length === 0 ? (
          <Text type="secondary" style={{ fontSize: 13 }}>
            No versions found
          </Text>
        ) : (
          displayVersions.map((version) => {
            const isActive = version.policy_id === currentPolicyId;
            const versionNumber = version.version_number ?? 1;
            const status = version.version_status ?? "draft";
            return (
              <div
                key={version.policy_id}
                className={`p-3 rounded-lg border transition-all cursor-pointer ${
                  isActive
                    ? "bg-blue-50 border-blue-300 shadow-sm"
                    : "bg-white border-gray-200 hover:border-gray-300"
                }`}
                onClick={() => onVersionSelect(version)}
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Text strong style={{ fontSize: 14 }}>
                      v{versionNumber}
                    </Text>
                    {isActive && <CheckCircleIcon className="w-4 h-4 text-blue-500" />}
                  </div>
                  <VersionStatusBadge status={status as "draft" | "published" | "production"} size="xs" />
                </div>
                <div className="flex items-center gap-1 mb-2">
                  <ClockIcon className="w-3 h-3 text-gray-400" />
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    {formatDate(version.created_at)}
                  </Text>
                </div>
                {version.description && (
                  <Text
                    type="secondary"
                    style={{ fontSize: 12, display: "block", marginBottom: 8 }}
                    className="line-clamp-2"
                  >
                    {version.description}
                  </Text>
                )}
                <div className="flex justify-end">{getActionButtons(version)}</div>
              </div>
            );
          })
        )}
      </div>
      <Divider style={{ margin: "16px 0" }} />
      {!isDemoMode && displayVersions.length >= 2 && (
        <div className="mb-4">
          <Button
            size="xs"
            variant="secondary"
            icon={SwitchHorizontalIcon}
            onClick={() => {
              setComparePolicyId1(currentPolicyId);
              setComparePolicyId2(
                displayVersions.find((v) => v.policy_id !== currentPolicyId)?.policy_id ?? ""
              );
              setCompareModalOpen(true);
            }}
          >
            Compare versions
          </Button>
        </div>
      )}
      {isDemoMode && (
        <Text type="secondary" style={{ fontSize: 12 }}>
          Demo – switch versions and use New Version. Changes are not saved.
        </Text>
      )}
      <Modal
        title="Compare versions"
        open={compareModalOpen}
        onCancel={() => {
          setCompareModalOpen(false);
          setComparePolicyId1("");
          setComparePolicyId2("");
        }}
        footer={null}
        width={720}
        destroyOnClose
      >
        <Form layout="vertical" className="mb-4">
          <div className="grid grid-cols-2 gap-4">
            <Form.Item label="Version A">
              <Select
                placeholder="Select version"
                value={comparePolicyId1 || undefined}
                onChange={setComparePolicyId1}
                style={{ width: "100%" }}
                options={displayVersions.map((v) => ({
                  label: `v${v.version_number ?? 1} (${v.version_status ?? "draft"})`,
                  value: v.policy_id,
                }))}
              />
            </Form.Item>
            <Form.Item label="Version B">
              <Select
                placeholder="Select version"
                value={comparePolicyId2 || undefined}
                onChange={setComparePolicyId2}
                style={{ width: "100%" }}
                options={displayVersions.map((v) => ({
                  label: `v${v.version_number ?? 1} (${v.version_status ?? "draft"})`,
                  value: v.policy_id,
                }))}
              />
            </Form.Item>
          </div>
        </Form>
        {comparePolicyId1 && comparePolicyId2 && comparePolicyId1 !== comparePolicyId2 && !isDemoMode && (
          <div style={{ maxHeight: "60vh", overflowY: "auto" }}>
            <VersionComparison
              policyId1={comparePolicyId1}
              policyId2={comparePolicyId2}
              accessToken={accessToken}
            />
          </div>
        )}
        {comparePolicyId1 && comparePolicyId2 && comparePolicyId1 === comparePolicyId2 && (
          <Text type="secondary">Select two different versions to compare.</Text>
        )}
      </Modal>
    </div>
  );
};

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
  isAdmin: boolean;
  onGuardrailUpdated?: () => void;
  onVersionSelect?: (version: Policy) => void;
  onVersionCreated?: () => void;
}

export const FlowBuilderPage: React.FC<FlowBuilderPageProps> = ({
  onBack,
  onSuccess,
  accessToken,
  editingPolicy,
  availableGuardrails,
  createPolicy,
  updatePolicy,
  isAdmin,
  onGuardrailUpdated,
  onVersionSelect,
  onVersionCreated,
}) => {
  const isDemoMode = editingPolicy == null;
  const [demoVersions, setDemoVersions] = useState<Policy[]>(() => [...MOCK_DEMO_VERSIONS]);
  const [activeDemoVersion, setActiveDemoVersion] = useState<Policy>(() => MOCK_DEMO_VERSIONS[0]);

  const effectivePolicy = isDemoMode ? activeDemoVersion : editingPolicy!;
  const isEditing = !!effectivePolicy?.policy_id && !isDemoMode;

  const [policyName, setPolicyName] = useState(effectivePolicy?.policy_name || "");
  const [description, setDescription] = useState(effectivePolicy?.description || "");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showTestPanel, setShowTestPanel] = useState(false);
  const [pipeline, setPipeline] = useState<GuardrailPipeline>(() =>
    getInitialPipelineFromPolicy(effectivePolicy)
  );

  // Sync local state when effective policy changes (switch version or real policy change)
  useEffect(() => {
    if (effectivePolicy) {
      setPolicyName(effectivePolicy.policy_name || "");
      setDescription(effectivePolicy.description || "");
      setPipeline(getInitialPipelineFromPolicy(effectivePolicy));
    }
  }, [effectivePolicy?.policy_id]);

  const handleDemoVersionSelect = (version: Policy) => {
    setActiveDemoVersion(version);
  };

  const handleDemoVersionCreated = () => {
    const nextNum = demoVersions.length + 1;
    const newVersion = createDemoVersion(
      `demo-v${nextNum}-${Date.now()}`,
      nextNum,
      "draft",
      pipeline?.steps ?? [createDefaultStep()]
    );
    setDemoVersions((prev) => [...prev, newVersion]);
    setActiveDemoVersion(newVersion);
  };

  const handleSave = async () => {
    if (isDemoMode) {
      message.info("Demo mode – changes are not saved. Use a real policy to save.");
      return;
    }
    if (!policyName.trim()) {
      message.error("Please enter a policy name");
      return;
    }
    if (!accessToken) {
      message.error("No access token available");
      return;
    }

    const steps = Array.isArray(pipeline?.steps) ? pipeline.steps : [];
    const emptySteps = steps.filter((s) => !s.guardrail);
    if (emptySteps.length > 0) {
      message.error("Please select a guardrail for all steps");
      return;
    }

    setIsSubmitting(true);
    try {
      const guardrailsFromPipeline = steps
        .map((s) => s.guardrail)
        .filter(Boolean);

      const data: PolicyCreateRequest | PolicyUpdateRequest = {
        policy_name: policyName,
        description: description || undefined,
        guardrails_add: guardrailsFromPipeline,
        guardrails_remove: [],
        pipeline: { mode: pipeline?.mode ?? "pre_call", steps },
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
        zIndex: 1100,
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
        pointerEvents: "auto",
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
            disabled={isEditing || isDemoMode}
            style={{ width: 240 }}
          />
          {(isEditing || isDemoMode) && effectivePolicy && (
            <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{ fontSize: 13, color: "#6b7280" }}>
                v{effectivePolicy.version_number ?? 1}
              </span>
              <VersionStatusBadge
                status={(effectivePolicy.version_status as "draft" | "published" | "production") ?? "draft"}
                size="xs"
              />
            </span>
          )}
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
          <Button
            variant="secondary"
            onClick={() => setShowTestPanel(!showTestPanel)}
          >
            {showTestPanel ? "Hide Test" : "Test Pipeline"}
          </Button>
          <Button
            onClick={handleSave}
            loading={isSubmitting}
            disabled={isDemoMode}
            title={isDemoMode ? "Demo – changes are not saved" : undefined}
          >
            {isDemoMode ? "Save disabled (demo)" : isEditing ? "Update Policy" : "Save Policy"}
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
          minWidth: 0,
        }}
      >
        <TextInput
          placeholder="Add a description (optional)..."
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          style={{ maxWidth: "100%", width: 560 }}
        />
      </div>

      {/* Flow builder canvas + test panel (with version sidebar when editing or demo) */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden", minHeight: 0 }}>
        {(isDemoMode || (isEditing && editingPolicy && onVersionSelect && onVersionCreated)) && (
          <div
            style={{
              width: 260,
              flexShrink: 0,
              borderRight: "1px solid #e5e7eb",
              backgroundColor: "#fff",
              overflowY: "auto",
            }}
          >
            <FlowBuilderVersionSidebar
              editingPolicy={effectivePolicy}
              accessToken={accessToken}
              onVersionSelect={isDemoMode ? handleDemoVersionSelect : onVersionSelect!}
              onVersionCreated={isDemoMode ? handleDemoVersionCreated : onVersionCreated!}
              versionsOverride={isDemoMode ? demoVersions : null}
            />
          </div>
        )}
        <div
          style={{
            flex: 1,
            minHeight: 0,
            minWidth: 400,
            overflowX: "auto",
            overflowY: "auto",
            display: "flex",
            justifyContent: "center",
            padding: "32px 24px",
          }}
        >
          <div style={{ minWidth: 760, width: "100%", maxWidth: 760, flexShrink: 0 }}>
            <PipelineFlowBuilder
              pipeline={pipeline}
              onChange={setPipeline}
              availableGuardrails={availableGuardrails}
              accessToken={accessToken}
              isAdmin={isAdmin}
              onGuardrailUpdated={onGuardrailUpdated}
            />
          </div>
        </div>

        {showTestPanel && (
          <PipelineTestPanel
            pipeline={pipeline}
            accessToken={accessToken}
            onClose={() => setShowTestPanel(false)}
          />
        )}
      </div>
    </div>
  );
};

export { createDefaultStep };
export default PipelineFlowBuilder;
