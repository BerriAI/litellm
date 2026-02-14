import React from "react";
import { Select, Typography, Tag } from "antd";
import { TextInput } from "@tremor/react";
import { PlusIcon, TrashIcon } from "@heroicons/react/outline";
import { GuardrailPipeline, PipelineStep } from "./types";
import { Guardrail } from "../guardrails/types";

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
  <div className="flex flex-col items-center" style={{ height: 48 }}>
    <div
      style={{
        width: 2,
        flex: 1,
        backgroundColor: "#d9d9d9",
      }}
    />
    <button
      onClick={onInsert}
      className="flex items-center justify-center"
      style={{
        width: 28,
        height: 28,
        borderRadius: "50%",
        border: "2px solid #d9d9d9",
        backgroundColor: "#fff",
        cursor: "pointer",
        zIndex: 1,
        marginTop: -4,
        marginBottom: -4,
      }}
      title="Insert step"
    >
      <PlusIcon style={{ width: 14, height: 14, color: "#8c8c8c" }} />
    </button>
    <div
      style={{
        width: 2,
        flex: 1,
        backgroundColor: "#d9d9d9",
      }}
    />
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
        border: "1px solid #e8e8e8",
        borderRadius: 12,
        padding: 20,
        backgroundColor: "#fff",
        boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
        maxWidth: 520,
        width: "100%",
      }}
    >
      {/* Header */}
      <div className="flex justify-between items-center" style={{ marginBottom: 16 }}>
        <div className="flex items-center gap-2">
          <Tag color="purple" style={{ margin: 0, fontWeight: 600, fontSize: 11 }}>
            GUARDRAIL
          </Tag>
          <Text strong>{step.guardrail || "Select guardrail..."}</Text>
        </div>
        <div className="flex items-center gap-2">
          <Text type="secondary" style={{ fontSize: 12 }}>
            Step {stepIndex + 1}
          </Text>
          <button
            onClick={onDelete}
            disabled={totalSteps <= 1}
            style={{
              background: "none",
              border: "none",
              cursor: totalSteps <= 1 ? "not-allowed" : "pointer",
              opacity: totalSteps <= 1 ? 0.3 : 1,
              padding: 4,
            }}
            title="Delete step"
          >
            <TrashIcon style={{ width: 16, height: 16, color: "#ff4d4f" }} />
          </button>
        </div>
      </div>

      {/* Guardrail Selector */}
      <div style={{ marginBottom: 12 }}>
        <Text type="secondary" style={{ fontSize: 12, display: "block", marginBottom: 4 }}>
          Guardrail
        </Text>
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

      {/* ON PASS */}
      <div
        style={{
          backgroundColor: "#f6ffed",
          border: "1px solid #b7eb8f",
          borderRadius: 8,
          padding: 12,
          marginBottom: 8,
        }}
      >
        <Text style={{ fontSize: 12, fontWeight: 600, color: "#52c41a" }}>
          ON PASS
        </Text>
        <Select
          style={{ width: "100%", marginTop: 4 }}
          value={step.on_pass}
          onChange={(value) => onChange({ on_pass: value as PipelineStep["on_pass"] })}
          options={ACTION_OPTIONS}
        />
        {step.on_pass === "modify_response" && (
          <div style={{ marginTop: 8 }}>
            <TextInput
              placeholder="Custom response message"
              value={step.modify_response_message || ""}
              onChange={(e) => onChange({ modify_response_message: e.target.value || null })}
            />
          </div>
        )}
      </div>

      {/* ON FAIL */}
      <div
        style={{
          backgroundColor: "#fff2f0",
          border: "1px solid #ffccc7",
          borderRadius: 8,
          padding: 12,
        }}
      >
        <Text style={{ fontSize: 12, fontWeight: 600, color: "#ff4d4f" }}>
          ON FAIL
        </Text>
        <Select
          style={{ width: "100%", marginTop: 4 }}
          value={step.on_fail}
          onChange={(value) => onChange({ on_fail: value as PipelineStep["on_fail"] })}
          options={ACTION_OPTIONS}
        />
        {step.on_fail === "modify_response" && (
          <div style={{ marginTop: 8 }}>
            <TextInput
              placeholder="Custom response message"
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
          border: "1px solid #d9d9d9",
          borderRadius: 12,
          padding: "16px 24px",
          backgroundColor: "#fafafa",
          maxWidth: 520,
          width: "100%",
        }}
      >
        <div className="flex items-center gap-3">
          <div
            style={{
              width: 32,
              height: 32,
              borderRadius: "50%",
              backgroundColor: "#f0f0f0",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <span style={{ fontSize: 16 }}>&#9654;</span>
          </div>
          <div>
            <Text type="secondary" style={{ fontSize: 11, fontWeight: 600, display: "block" }}>
              TRIGGER
            </Text>
            <Text strong>Incoming LLM Request</Text>
            <br />
            <Text type="secondary" style={{ fontSize: 12 }}>
              This flow runs when a request matches this policy
            </Text>
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

const ACTION_COLORS: Record<string, string> = {
  allow: "green",
  block: "red",
  next: "blue",
  modify_response: "orange",
};

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
  <div className="flex flex-col items-center" style={{ padding: "16px 0" }}>
    {/* Trigger */}
    <div
      style={{
        border: "1px solid #d9d9d9",
        borderRadius: 12,
        padding: "12px 20px",
        backgroundColor: "#fafafa",
        maxWidth: 520,
        width: "100%",
      }}
    >
      <div className="flex items-center gap-3">
        <span style={{ fontSize: 16 }}>&#9654;</span>
        <div>
          <Text type="secondary" style={{ fontSize: 11, fontWeight: 600 }}>TRIGGER</Text>
          <br />
          <Text strong>Incoming LLM Request</Text>
        </div>
      </div>
    </div>

    {/* Steps */}
    {pipeline.steps.map((step, index) => (
      <React.Fragment key={index}>
        {/* Connector line */}
        <div style={{ width: 2, height: 32, backgroundColor: "#d9d9d9" }} />

        {/* Step card */}
        <div
          style={{
            border: "1px solid #e8e8e8",
            borderRadius: 12,
            padding: "12px 20px",
            backgroundColor: "#fff",
            maxWidth: 520,
            width: "100%",
          }}
        >
          <div className="flex justify-between items-center" style={{ marginBottom: 8 }}>
            <div className="flex items-center gap-2">
              <Tag color="purple" style={{ margin: 0, fontSize: 11, fontWeight: 600 }}>
                GUARDRAIL
              </Tag>
              <Text strong>{step.guardrail}</Text>
            </div>
            <Text type="secondary" style={{ fontSize: 12 }}>Step {index + 1}</Text>
          </div>
          <div className="flex gap-3">
            <Text type="secondary" style={{ fontSize: 12 }}>
              Pass →{" "}
              <Tag color={ACTION_COLORS[step.on_pass]} style={{ fontSize: 11 }}>
                {ACTION_LABELS[step.on_pass] || step.on_pass}
              </Tag>
            </Text>
            <Text type="secondary" style={{ fontSize: 12 }}>
              Fail →{" "}
              <Tag color={ACTION_COLORS[step.on_fail]} style={{ fontSize: 11 }}>
                {ACTION_LABELS[step.on_fail] || step.on_fail}
              </Tag>
            </Text>
          </div>
        </div>
      </React.Fragment>
    ))}
  </div>
);

export { createDefaultStep };
export default PipelineFlowBuilder;
