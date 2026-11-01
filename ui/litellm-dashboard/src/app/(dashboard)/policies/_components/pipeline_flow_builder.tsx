import React, { useState } from "react";
import type { ParseKeys, TFunction } from "i18next";
import { useTranslation } from "react-i18next";
import { toast } from "@/lib/toast";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { ArrowLeft, MoreVertical, Plus } from "lucide-react";
import {
  GuardrailPipeline,
  PipelineStep,
  PipelineTestResult,
  PolicyCreateRequest,
  PolicyUpdateRequest,
  Policy,
} from "@/components/policies/types";
import { Guardrail } from "@/components/guardrails/types";
import {
  testPipelineCall,
  listPolicyVersions,
  createPolicyVersion,
  updatePolicyVersionStatus,
} from "@/components/networking";
import { getComplianceDatasetPrompts, getFrameworks } from "@/data/compliancePrompts";
import type { CompliancePrompt } from "@/data/compliancePrompts";

const TEST_SOURCE_QUICK = "quick_chat";
const TEST_SOURCE_ALL = "__all__";

function getPromptsForTestSource(source: string): CompliancePrompt[] {
  if (source === TEST_SOURCE_QUICK) return [];
  if (source === TEST_SOURCE_ALL) return getComplianceDatasetPrompts();
  const fw = getFrameworks().find((f) => f.name === source);
  return fw ? fw.categories.flatMap((c) => c.prompts) : [];
}

const ACTION_OPTIONS: { labelKey: ParseKeys; value: string }[] = [
  { labelKey: "policies.flowBuilder.actions.next", value: "next" },
  { labelKey: "policies.flowBuilder.actions.allow", value: "allow" },
  { labelKey: "policies.flowBuilder.actions.block", value: "block" },
  { labelKey: "policies.flowBuilder.actions.modifyResponse", value: "modify_response" },
];

const ACTION_LABEL_KEYS: Record<string, ParseKeys> = {
  allow: "policies.flowBuilder.actions.allow",
  block: "policies.flowBuilder.actions.block",
  next: "policies.flowBuilder.actions.next",
  modify_response: "policies.flowBuilder.actions.modifyResponse",
};

function actionLabel(action: string, t: TFunction): string {
  const key = ACTION_LABEL_KEYS[action];
  return key ? t(key) : action;
}

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

function updateStepAtIndex(steps: PipelineStep[], index: number, updated: Partial<PipelineStep>): PipelineStep[] {
  return steps.map((s, i) => (i === index ? { ...s, ...updated } : s));
}

/**
 * Derives a pipeline from a policy. When the policy has a pipeline, use it.
 * When it only has guardrails_add (legacy/simple form), convert those guardrails
 * into pipeline steps in order.
 */
function derivePipelineFromPolicy(policy: Policy | null | undefined): GuardrailPipeline {
  if (!policy) {
    return { mode: "pre_call", steps: [createDefaultStep()] };
  }
  if (policy.pipeline?.steps?.length) {
    return policy.pipeline;
  }
  const guardrails = policy.guardrails_add || [];
  if (guardrails.length > 0) {
    return {
      mode: policy.pipeline?.mode ?? "pre_call",
      steps: guardrails.map((g) => ({
        guardrail: g,
        on_pass: "next" as const,
        on_fail: "block" as const,
        pass_data: false,
        modify_response_message: null,
      })),
    };
  }
  return { mode: "pre_call", steps: [createDefaultStep()] };
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
      backgroundColor: "color-mix(in oklab, var(--color-info) 10%, transparent)",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      flexShrink: 0,
    }}
  >
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      style={{ color: "var(--color-info)" }}
      strokeLinecap="round"
      strokeLinejoin="round"
    >
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
      backgroundColor: "var(--color-muted)",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      flexShrink: 0,
    }}
  >
    <svg
      width="12"
      height="12"
      viewBox="0 0 24 24"
      fill="currentColor"
      stroke="none"
      style={{ color: "var(--color-muted-foreground)" }}
    >
      <polygon points="6,3 20,12 6,21" />
    </svg>
  </div>
);

const PassIcon: React.FC = () => (
  <svg
    width="14"
    height="14"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2.5"
    strokeLinecap="round"
    strokeLinejoin="round"
    style={{ flexShrink: 0, color: "var(--color-success)" }}
  >
    <circle cx="12" cy="12" r="10" />
    <path d="M9 12l2 2 4-4" />
  </svg>
);

const FailIcon: React.FC = () => (
  <svg
    width="14"
    height="14"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2.5"
    strokeLinecap="round"
    strokeLinejoin="round"
    style={{ flexShrink: 0, color: "var(--color-destructive)" }}
  >
    <circle cx="12" cy="12" r="10" />
  </svg>
);

const ApiFailureIcon: React.FC = () => (
  <svg
    width="14"
    height="14"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    style={{ flexShrink: 0, color: "var(--color-warning)" }}
  >
    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
    <line x1="12" y1="9" x2="12" y2="13" />
    <line x1="12" y1="17" x2="12.01" y2="17" />
  </svg>
);

// ─────────────────────────────────────────────────────────────────────────────
// Connector
// ─────────────────────────────────────────────────────────────────────────────

interface ConnectorProps {
  onInsert: () => void;
}

const Connector: React.FC<ConnectorProps> = ({ onInsert }) => {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col items-center" style={{ height: 56 }}>
      <div style={{ width: 1, flex: 1, backgroundColor: "var(--color-border)" }} />
      <button
        onClick={onInsert}
        className="z-raised flex items-center justify-center"
        style={{
          width: 24,
          height: 24,
          borderRadius: "50%",
          border: "1px solid var(--color-border)",
          backgroundColor: "var(--color-card)",
          cursor: "pointer",
          transition: "all 0.15s ease",
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.borderColor = "var(--color-info)";
          e.currentTarget.style.backgroundColor = "color-mix(in oklab, var(--color-info) 10%, transparent)";
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.borderColor = "var(--color-border)";
          e.currentTarget.style.backgroundColor = "var(--color-card)";
        }}
        title={t("policies.flowBuilder.connector.insertStep")}
      >
        <Plus style={{ width: 12, height: 12, color: "var(--color-muted-foreground)" }} />
      </button>
      <div style={{ width: 1, flex: 1, backgroundColor: "var(--color-border)" }} />
    </div>
  );
};

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
  const { t } = useTranslation();
  const guardrailOptions = availableGuardrails.map((g) => ({
    label: g.guardrail_name || g.guardrail_id,
    value: g.guardrail_name || g.guardrail_id,
  }));

  return (
    <div
      style={{
        border: "1px solid var(--color-border)",
        borderRadius: 10,
        backgroundColor: "var(--color-card)",
        maxWidth: 720,
        width: "100%",
        overflow: "hidden",
      }}
    >
      {/* Header row */}
      <div className="flex items-center justify-between" style={{ padding: "14px 20px 0 20px" }}>
        <div className="flex items-center gap-2">
          <GuardrailIcon />
          <span
            style={{
              fontSize: 11,
              fontWeight: 700,
              textTransform: "uppercase",
              color: "var(--color-info)",
              letterSpacing: "0.06em",
            }}
          >
            {t("policies.flowBuilder.stepCard.typeLabel")}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span style={{ fontSize: 13, color: "var(--color-muted-foreground)" }}>
            {t("policies.flowBuilder.stepCard.stepNumber", { number: stepIndex + 1 })}
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
            title={t("policies.flowBuilder.stepCard.deleteStep")}
          >
            <MoreVertical style={{ width: 16, height: 16, color: "var(--color-muted-foreground)" }} />
          </button>
        </div>
      </div>

      {/* Guardrail selector */}
      <div style={{ padding: "12px 20px 16px 20px" }}>
        <label
          style={{
            fontSize: 12,
            fontWeight: 500,
            color: "var(--color-muted-foreground)",
            display: "block",
            marginBottom: 6,
          }}
        >
          {t("policies.flowBuilder.stepCard.guardrailLabel")}
        </label>
        <SearchSelect
          options={guardrailOptions}
          value={step.guardrail || undefined}
          onValueChange={(value) => onChange({ guardrail: value ?? undefined })}
          placeholder={t("policies.flowBuilder.stepCard.guardrailPlaceholder")}
          emptyText={t("policies.flowBuilder.stepCard.noGuardrails")}
        />
      </div>

      {/* ON PASS section */}
      <div style={{ borderTop: "1px solid var(--color-border)", padding: "14px 20px" }}>
        <div className="flex items-center gap-2" style={{ marginBottom: 8 }}>
          <PassIcon />
          <span style={{ fontSize: 13, fontWeight: 600, color: "var(--color-foreground)" }}>
            {t("policies.flowBuilder.stepCard.onPass")}
          </span>
        </div>
        <label
          style={{
            fontSize: 12,
            fontWeight: 500,
            color: "var(--color-muted-foreground)",
            display: "block",
            marginBottom: 6,
          }}
        >
          {t("policies.flowBuilder.stepCard.actionLabel")}
        </label>
        <Select value={step.on_pass} onValueChange={(value) => onChange({ on_pass: value as PipelineStep["on_pass"] })}>
          <SelectTrigger className="w-full">
            <SelectValue>{actionLabel(step.on_pass, t)}</SelectValue>
          </SelectTrigger>
          <SelectContent>
            {ACTION_OPTIONS.map((option) => (
              <SelectItem key={option.value} value={option.value}>
                {t(option.labelKey)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {step.on_pass === "modify_response" && (
          <div style={{ marginTop: 8 }}>
            <label
              style={{
                fontSize: 12,
                fontWeight: 500,
                color: "var(--color-muted-foreground)",
                display: "block",
                marginBottom: 6,
              }}
            >
              {t("policies.flowBuilder.stepCard.customResponseLabel")}
            </label>
            <Input
              placeholder={t("policies.flowBuilder.stepCard.customResponsePlaceholder")}
              value={step.modify_response_message || ""}
              onChange={(e) => onChange({ modify_response_message: e.target.value || null })}
            />
          </div>
        )}
      </div>

      {/* ON FAIL section */}
      <div style={{ borderTop: "1px solid var(--color-border)", padding: "14px 20px" }}>
        <div className="flex items-center gap-2" style={{ marginBottom: 8 }}>
          <FailIcon />
          <span style={{ fontSize: 13, fontWeight: 600, color: "var(--color-foreground)" }}>
            {t("policies.flowBuilder.stepCard.onFail")}
          </span>
        </div>
        <label
          style={{
            fontSize: 12,
            fontWeight: 500,
            color: "var(--color-muted-foreground)",
            display: "block",
            marginBottom: 6,
          }}
        >
          {t("policies.flowBuilder.stepCard.actionLabel")}
        </label>
        <Select value={step.on_fail} onValueChange={(value) => onChange({ on_fail: value as PipelineStep["on_fail"] })}>
          <SelectTrigger className="w-full">
            <SelectValue>{actionLabel(step.on_fail, t)}</SelectValue>
          </SelectTrigger>
          <SelectContent>
            {ACTION_OPTIONS.map((option) => (
              <SelectItem key={option.value} value={option.value}>
                {t(option.labelKey)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {step.on_fail === "modify_response" && (
          <div style={{ marginTop: 8 }}>
            <label
              style={{
                fontSize: 12,
                fontWeight: 500,
                color: "var(--color-muted-foreground)",
                display: "block",
                marginBottom: 6,
              }}
            >
              {t("policies.flowBuilder.stepCard.customResponseLabel")}
            </label>
            <Input
              placeholder={t("policies.flowBuilder.stepCard.customResponsePlaceholder")}
              value={step.modify_response_message || ""}
              onChange={(e) => onChange({ modify_response_message: e.target.value || null })}
            />
          </div>
        )}
      </div>

      {/* ON API FAILURE (technical / provider outage) — optional; defaults to ON FAIL */}
      <div style={{ borderTop: "1px solid var(--color-border)", padding: "14px 20px" }}>
        <div className="flex items-center gap-2" style={{ marginBottom: 8 }}>
          <ApiFailureIcon />
          <span style={{ fontSize: 13, fontWeight: 600, color: "var(--color-foreground)" }}>
            {t("policies.flowBuilder.stepCard.onApiFailure")}
          </span>
        </div>
        <label
          style={{
            fontSize: 12,
            fontWeight: 500,
            color: "var(--color-muted-foreground)",
            display: "block",
            marginBottom: 6,
          }}
        >
          {t("policies.flowBuilder.stepCard.actionLabel")}
        </label>
        <Select
          value={step.on_error ?? null}
          onValueChange={(value) =>
            onChange({ on_error: value === null ? undefined : (value as PipelineStep["on_error"]) })
          }
        >
          <SelectTrigger className="w-full">
            <SelectValue>
              {step.on_error != null ? actionLabel(step.on_error, t) : t("policies.flowBuilder.stepCard.sameAsOnFail")}
            </SelectValue>
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={null}>{t("policies.flowBuilder.stepCard.sameAsOnFail")}</SelectItem>
            {ACTION_OPTIONS.map((option) => (
              <SelectItem key={option.value} value={option.value}>
                {t(option.labelKey)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {step.on_error === "modify_response" && step.on_fail !== "modify_response" && (
          <div style={{ marginTop: 8 }}>
            <label
              style={{
                fontSize: 12,
                fontWeight: 500,
                color: "var(--color-muted-foreground)",
                display: "block",
                marginBottom: 6,
              }}
            >
              {t("policies.flowBuilder.stepCard.customResponseLabel")}
            </label>
            <Input
              placeholder={t("policies.flowBuilder.stepCard.customResponsePlaceholder")}
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

const PipelineFlowBuilder: React.FC<PipelineFlowBuilderProps> = ({ pipeline, onChange, availableGuardrails }) => {
  const { t } = useTranslation();
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
          border: "1px solid var(--color-border)",
          borderRadius: 10,
          padding: "16px 20px",
          backgroundColor: "var(--color-card)",
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
                color: "var(--color-muted-foreground)",
                letterSpacing: "0.06em",
                display: "block",
                marginBottom: 2,
              }}
            >
              {t("policies.flowBuilder.canvas.trigger.label")}
            </span>
            <span style={{ fontSize: 14, fontWeight: 600, color: "var(--color-foreground)", display: "block" }}>
              {t("policies.flowBuilder.canvas.trigger.title")}
            </span>
            <span style={{ fontSize: 13, color: "var(--color-muted-foreground)" }}>
              {t("policies.flowBuilder.canvas.trigger.description")}
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

      {/* End card */}
      <div
        style={{
          border: "1px solid var(--color-border)",
          borderRadius: 10,
          padding: "14px 20px",
          backgroundColor: "var(--color-card)",
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
              backgroundColor: "var(--color-muted)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexShrink: 0,
            }}
          >
            <svg
              width="12"
              height="12"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              style={{ color: "var(--color-muted-foreground)" }}
            >
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
                color: "var(--color-muted-foreground)",
                letterSpacing: "0.06em",
                display: "block",
                marginBottom: 2,
              }}
            >
              {t("policies.flowBuilder.canvas.end.label")}
            </span>
            <span style={{ fontSize: 14, fontWeight: 600, color: "var(--color-foreground)", display: "block" }}>
              {t("policies.flowBuilder.canvas.end.title")}
            </span>
            <span style={{ fontSize: 13, color: "var(--color-muted-foreground)" }}>
              {t("policies.flowBuilder.canvas.end.description")}
            </span>
          </div>
        </div>
      </div>
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
  const { t } = useTranslation();
  return (
    <div className="flex flex-col items-center" style={{ padding: "16px 0" }}>
      {/* Trigger */}
      <div
        style={{
          border: "1px solid var(--color-border)",
          borderRadius: 10,
          padding: "14px 20px",
          backgroundColor: "var(--color-card)",
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
                color: "var(--color-muted-foreground)",
                letterSpacing: "0.06em",
                display: "block",
                marginBottom: 2,
              }}
            >
              {t("policies.flowBuilder.infoDisplay.triggerLabel")}
            </span>
            <span style={{ fontSize: 14, fontWeight: 600, color: "var(--color-foreground)" }}>
              {t("policies.flowBuilder.infoDisplay.triggerTitle")}
            </span>
          </div>
        </div>
      </div>

      {/* Steps */}
      {pipeline.steps.map((step, index) => (
        <React.Fragment key={index}>
          {/* Connector */}
          <div style={{ width: 1, height: 32, backgroundColor: "var(--color-border)" }} />

          {/* Step card */}
          <div
            style={{
              border: "1px solid var(--color-border)",
              borderRadius: 10,
              padding: "14px 20px",
              backgroundColor: "var(--color-card)",
              maxWidth: 720,
              width: "100%",
            }}
          >
            {/* Header */}
            <div className="flex items-center justify-between" style={{ marginBottom: 8 }}>
              <div className="flex items-center gap-2">
                <GuardrailIcon />
                <span
                  style={{
                    fontSize: 11,
                    fontWeight: 700,
                    textTransform: "uppercase",
                    color: "var(--color-info)",
                    letterSpacing: "0.06em",
                  }}
                >
                  {t("policies.flowBuilder.infoDisplay.typeLabel")}
                </span>
              </div>
              <span style={{ fontSize: 13, color: "var(--color-muted-foreground)" }}>
                {t("policies.flowBuilder.infoDisplay.stepNumber", { number: index + 1 })}
              </span>
            </div>

            {/* Name */}
            <div style={{ fontSize: 15, fontWeight: 600, color: "var(--color-foreground)", marginBottom: 8 }}>
              {step.guardrail}
            </div>

            {/* Divider */}
            <div style={{ borderTop: "1px solid var(--color-muted)", marginBottom: 10 }} />

            {/* Pass / Fail / API failure */}
            <div className="flex flex-col gap-2" style={{ fontSize: 13, color: "var(--color-foreground)" }}>
              <span className="flex items-center gap-1.5">
                <PassIcon />{" "}
                {t("policies.flowBuilder.infoDisplay.passAction", { action: actionLabel(step.on_pass, t) })}
              </span>
              <span className="flex items-center gap-1.5">
                <FailIcon />{" "}
                {t("policies.flowBuilder.infoDisplay.failAction", { action: actionLabel(step.on_fail, t) })}
              </span>
              <span className="flex items-center gap-1.5">
                <ApiFailureIcon />{" "}
                {t("policies.flowBuilder.infoDisplay.apiFailureAction", {
                  action:
                    step.on_error != null
                      ? actionLabel(step.on_error, t)
                      : t("policies.flowBuilder.infoDisplay.sameAsOnFail", { action: actionLabel(step.on_fail, t) }),
                })}
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

const OUTCOME_STYLES: Record<string, { bg: string; color: string; labelKey: ParseKeys }> = {
  pass: {
    bg: "color-mix(in oklab, var(--color-success) 10%, transparent)",
    color: "var(--color-success)",
    labelKey: "policies.flowBuilder.testPanel.outcomes.pass",
  },
  fail: {
    bg: "color-mix(in oklab, var(--color-destructive) 10%, transparent)",
    color: "var(--color-destructive)",
    labelKey: "policies.flowBuilder.testPanel.outcomes.fail",
  },
  error: {
    bg: "color-mix(in oklab, var(--color-warning) 10%, transparent)",
    color: "var(--color-warning)",
    labelKey: "policies.flowBuilder.testPanel.outcomes.error",
  },
};

const TERMINAL_STYLES: Record<string, { bg: string; color: string }> = {
  allow: { bg: "color-mix(in oklab, var(--color-success) 10%, transparent)", color: "var(--color-success)" },
  block: { bg: "color-mix(in oklab, var(--color-destructive) 10%, transparent)", color: "var(--color-destructive)" },
  modify_response: { bg: "color-mix(in oklab, var(--color-info) 10%, transparent)", color: "var(--color-info)" },
};

interface ComplianceRunEntry {
  prompt: CompliancePrompt;
  result: PipelineTestResult | null;
  error?: string;
  matched: boolean;
}

function complianceMatchExpected(expected: "pass" | "fail", terminalAction: string): boolean {
  if (expected === "pass") {
    return terminalAction === "allow" || terminalAction === "modify_response";
  }
  return terminalAction === "block";
}

type TestSourceOption = { value: string; labelKey: ParseKeys } | { value: string; label: string };

const testSourceOptions: TestSourceOption[] = [
  { value: TEST_SOURCE_QUICK, labelKey: "policies.flowBuilder.testPanel.sources.quickChat" },
  ...getFrameworks().map((f) => ({ value: f.name, label: f.name })),
  { value: TEST_SOURCE_ALL, labelKey: "policies.flowBuilder.testPanel.sources.allDatasets" },
];

function testSourceLabel(option: TestSourceOption, t: TFunction): string {
  return "labelKey" in option ? t(option.labelKey) : option.label;
}

const PipelineTestPanel: React.FC<PipelineTestPanelProps> = ({ pipeline, accessToken, onClose }) => {
  const { t } = useTranslation();
  const [testSource, setTestSource] = useState<string>(TEST_SOURCE_QUICK);
  const [testMessage, setTestMessage] = useState("Hello, can you help me?");
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<PipelineTestResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [complianceResults, setComplianceResults] = useState<ComplianceRunEntry[]>([]);

  const isQuickChat = testSource === TEST_SOURCE_QUICK;
  const promptsForSource = getPromptsForTestSource(testSource);
  const isDataset = promptsForSource.length > 0;
  const selectedSource = testSourceOptions.find((option) => option.value === testSource);

  const handleRunTest = async () => {
    if (!accessToken) return;

    const emptySteps = pipeline.steps.filter((s) => !s.guardrail);
    if (emptySteps.length > 0) {
      setError(t("policies.flowBuilder.testPanel.errors.missingGuardrail"));
      return;
    }

    setError(null);
    setIsRunning(true);
    setResult(null);
    setComplianceResults([]);

    if (isQuickChat) {
      try {
        const data = await testPipelineCall(accessToken, pipeline, [{ role: "user", content: testMessage }]);
        setResult(data);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setIsRunning(false);
      }
      return;
    }

    const entries: ComplianceRunEntry[] = [];
    for (const prompt of promptsForSource) {
      try {
        const data = await testPipelineCall(accessToken, pipeline, [{ role: "user", content: prompt.prompt }]);
        const matched = complianceMatchExpected(prompt.expectedResult, data.terminal_action);
        entries.push({ prompt, result: data, matched });
      } catch (e) {
        const errMsg = e instanceof Error ? e.message : String(e);
        entries.push({
          prompt,
          result: null,
          error: errMsg,
          matched: false,
        });
      }
    }
    setComplianceResults(entries);
    setIsRunning(false);
  };

  return (
    <div
      style={{
        width: 400,
        borderLeft: "1px solid var(--color-border)",
        backgroundColor: "var(--color-card)",
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
          borderBottom: "1px solid var(--color-border)",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <span style={{ fontSize: 14, fontWeight: 600, color: "var(--color-foreground)" }}>
          {t("policies.flowBuilder.testPanel.title")}
        </span>
        <button
          onClick={onClose}
          style={{
            background: "none",
            border: "none",
            cursor: "pointer",
            fontSize: 18,
            color: "var(--color-muted-foreground)",
            padding: "0 4px",
          }}
        >
          x
        </button>
      </div>

      {/* Input section */}
      <div style={{ padding: 16, borderBottom: "1px solid var(--color-border)" }}>
        <label
          style={{
            fontSize: 12,
            fontWeight: 500,
            color: "var(--color-muted-foreground)",
            display: "block",
            marginBottom: 6,
          }}
        >
          {t("policies.flowBuilder.testPanel.sourceLabel")}
        </label>
        <Select value={testSource} onValueChange={(value) => value !== null && setTestSource(value)}>
          <SelectTrigger className="mb-3 w-full">
            <SelectValue>{selectedSource ? testSourceLabel(selectedSource, t) : testSource}</SelectValue>
          </SelectTrigger>
          <SelectContent>
            {testSourceOptions.map((option) => (
              <SelectItem key={option.value} value={option.value}>
                {testSourceLabel(option, t)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {isQuickChat && (
          <>
            <label
              style={{
                fontSize: 12,
                fontWeight: 500,
                color: "var(--color-muted-foreground)",
                display: "block",
                marginBottom: 6,
              }}
            >
              {t("policies.flowBuilder.testPanel.messageLabel")}
            </label>
            <textarea
              value={testMessage}
              onChange={(e) => setTestMessage(e.target.value)}
              placeholder={t("policies.flowBuilder.testPanel.messagePlaceholder")}
              rows={3}
              style={{
                width: "100%",
                border: "1px solid var(--color-border)",
                borderRadius: 6,
                padding: "8px 10px",
                fontSize: 13,
                resize: "vertical",
                fontFamily: "inherit",
                backgroundColor: "var(--color-card)",
                color: "var(--color-foreground)",
              }}
            />
          </>
        )}
        {isDataset && (
          <div
            style={{
              fontSize: 12,
              color: "var(--color-muted-foreground)",
              padding: "8px 10px",
              backgroundColor: "var(--color-muted)",
              borderRadius: 6,
              marginBottom: 8,
            }}
          >
            {testSource === TEST_SOURCE_ALL
              ? t("policies.flowBuilder.testPanel.allDatasetsDescription")
              : t("policies.flowBuilder.testPanel.frameworkDescription", {
                  promptCount: promptsForSource.length,
                  source: testSource,
                })}
          </div>
        )}
        <Button onClick={handleRunTest} disabled={isRunning} style={{ marginTop: 8, width: "100%" }}>
          {t("policies.flowBuilder.testPanel.runTest")}
        </Button>
      </div>

      {/* Results section */}
      <div style={{ flex: 1, overflowY: "auto", padding: 16 }}>
        {error && (
          <div
            style={{
              padding: "10px 12px",
              backgroundColor: "color-mix(in oklab, var(--color-destructive) 10%, transparent)",
              border: "1px solid color-mix(in oklab, var(--color-destructive) 30%, transparent)",
              borderRadius: 6,
              fontSize: 13,
              color: "var(--color-destructive)",
              marginBottom: 12,
            }}
          >
            {error}
          </div>
        )}

        {result && (
          <div>
            {/* Step results */}
            {result.step_results.map((step, i) => {
              const style = OUTCOME_STYLES[step.outcome] || OUTCOME_STYLES.error;
              return (
                <div
                  key={i}
                  style={{
                    border: "1px solid var(--color-border)",
                    borderRadius: 8,
                    padding: "10px 12px",
                    marginBottom: 8,
                  }}
                >
                  <div className="flex items-center justify-between" style={{ marginBottom: 4 }}>
                    <span style={{ fontSize: 13, fontWeight: 600, color: "var(--color-foreground)" }}>
                      {t("policies.flowBuilder.testPanel.stepResult", {
                        number: i + 1,
                        guardrail: step.guardrail_name,
                      })}
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
                      {t(style.labelKey)}
                    </span>
                  </div>
                  <div style={{ fontSize: 12, color: "var(--color-muted-foreground)" }}>
                    {t("policies.flowBuilder.testPanel.stepAction", { action: actionLabel(step.action_taken, t) })}
                    {step.duration_seconds != null && (
                      <span style={{ marginLeft: 8 }}>
                        {t("policies.flowBuilder.testPanel.duration", {
                          ms: (step.duration_seconds * 1000).toFixed(0),
                        })}
                      </span>
                    )}
                  </div>
                  {step.error_detail && (
                    <div style={{ fontSize: 12, color: "var(--color-destructive)", marginTop: 4 }}>
                      {step.error_detail}
                    </div>
                  )}
                </div>
              );
            })}

            {/* Terminal result */}
            <div
              style={{
                borderTop: "1px solid var(--color-border)",
                paddingTop: 12,
                marginTop: 4,
              }}
            >
              <div className="flex items-center justify-between">
                <span style={{ fontSize: 13, fontWeight: 600, color: "var(--color-foreground)" }}>
                  {t("policies.flowBuilder.testPanel.result")}
                </span>
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
                      {result.terminal_action === "modify_response"
                        ? t("policies.flowBuilder.actions.modifyResponse")
                        : result.terminal_action}
                    </span>
                  );
                })()}
              </div>
              {result.error_message && (
                <div style={{ fontSize: 12, color: "var(--color-destructive)", marginTop: 6 }}>
                  {result.error_message}
                </div>
              )}
              {result.modify_response_message && (
                <div style={{ fontSize: 12, color: "var(--color-info)", marginTop: 6 }}>
                  {t("policies.flowBuilder.testPanel.customResponse", { message: result.modify_response_message })}
                </div>
              )}
            </div>
          </div>
        )}

        {complianceResults.length > 0 && (
          <div style={{ marginTop: 16 }}>
            <div
              style={{
                fontSize: 13,
                fontWeight: 600,
                color: "var(--color-foreground)",
                marginBottom: 8,
              }}
            >
              {t("policies.flowBuilder.testPanel.compliance.title")}
            </div>
            <div
              style={{
                fontSize: 12,
                color: "var(--color-muted-foreground)",
                marginBottom: 10,
              }}
            >
              {t("policies.flowBuilder.testPanel.compliance.matchedSummary", {
                matched: complianceResults.filter((e) => e.matched).length,
                total: complianceResults.length,
              })}
            </div>
            <div
              style={{
                maxHeight: 320,
                overflowY: "auto",
                border: "1px solid var(--color-border)",
                borderRadius: 8,
              }}
            >
              {complianceResults.map((entry, i) => {
                const actual = entry.result?.terminal_action ?? (entry.error ? "error" : "—");
                const matchStyle = entry.matched
                  ? { bg: "color-mix(in oklab, var(--color-success) 10%, transparent)", color: "var(--color-success)" }
                  : {
                      bg: "color-mix(in oklab, var(--color-destructive) 10%, transparent)",
                      color: "var(--color-destructive)",
                    };
                return (
                  <div
                    key={entry.prompt.id ?? i}
                    style={{
                      padding: "8px 10px",
                      borderBottom: i < complianceResults.length - 1 ? "1px solid var(--color-border)" : "none",
                      fontSize: 12,
                    }}
                  >
                    <div
                      style={{
                        color: "var(--color-foreground)",
                        marginBottom: 4,
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                      }}
                      title={entry.prompt.prompt}
                    >
                      {entry.prompt.prompt}
                    </div>
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 8,
                        flexWrap: "wrap",
                      }}
                    >
                      <span style={{ color: "var(--color-muted-foreground)" }}>
                        {t("policies.flowBuilder.testPanel.compliance.expected", {
                          result: entry.prompt.expectedResult,
                        })}
                      </span>
                      <span style={{ color: "var(--color-muted-foreground)" }}>→</span>
                      <span style={{ color: "var(--color-muted-foreground)" }}>
                        {t("policies.flowBuilder.testPanel.compliance.actual", { result: actual })}
                      </span>
                      <span
                        style={{
                          backgroundColor: matchStyle.bg,
                          color: matchStyle.color,
                          padding: "1px 6px",
                          borderRadius: 4,
                          fontWeight: 600,
                        }}
                      >
                        {entry.matched ? "✓" : "✗"}
                      </span>
                    </div>
                    {entry.error && (
                      <div style={{ color: "var(--color-destructive)", marginTop: 4 }}>{entry.error}</div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {!result && !error && complianceResults.length === 0 && (
          <div style={{ textAlign: "center", color: "var(--color-muted-foreground)", fontSize: 13, marginTop: 24 }}>
            {t("policies.flowBuilder.testPanel.emptyState")}
          </div>
        )}
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Policy Versions Sidebar (left sidebar when editing a policy)
// ─────────────────────────────────────────────────────────────────────────────

const VERSION_STATUS_STYLES: Record<string, { bg: string; color: string }> = {
  draft: { bg: "var(--color-muted)", color: "var(--color-muted-foreground)" },
  published: { bg: "color-mix(in oklab, var(--color-info) 10%, transparent)", color: "var(--color-info)" },
  production: { bg: "color-mix(in oklab, var(--color-success) 10%, transparent)", color: "var(--color-success)" },
};

interface PolicyVersionsSidebarProps {
  policyName: string;
  editingPolicyId: string | null;
  editingVersionStatus?: "draft" | "published" | "production";
  accessToken: string | null;
  versions: Policy[];
  isLoading: boolean;
  isCreatingVersion?: boolean;
  isUpdatingStatus?: boolean;
  onNewVersion: () => void;
  onSelectVersion: (policy: Policy) => void;
  onPublish?: () => void;
  onPromoteToProduction?: () => void;
}

const PolicyVersionsSidebar: React.FC<PolicyVersionsSidebarProps> = ({
  policyName,
  editingPolicyId,
  editingVersionStatus,
  accessToken,
  versions,
  isLoading,
  isCreatingVersion = false,
  isUpdatingStatus = false,
  onNewVersion,
  onSelectVersion,
  onPublish,
  onPromoteToProduction,
}) => {
  const { t } = useTranslation();
  const canPublish = editingVersionStatus === "draft" && onPublish;
  const canPromote = editingVersionStatus === "published" && onPromoteToProduction;

  return (
    <div
      style={{
        width: 260,
        flexShrink: 0,
        backgroundColor: "var(--color-card)",
        borderRight: "1px solid var(--color-border)",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
      }}
    >
      <div style={{ padding: 16, overflowY: "auto", flex: 1 }}>
        {/* Versions section */}
        <div style={{ marginBottom: 24 }}>
          <span
            style={{
              fontSize: 11,
              fontWeight: 700,
              textTransform: "uppercase",
              color: "var(--color-muted-foreground)",
              letterSpacing: "0.06em",
              display: "block",
              marginBottom: 4,
            }}
          >
            {t("policies.flowBuilder.versionsSidebar.title")}
          </span>
          <span
            style={{
              fontSize: 11,
              color: "var(--color-muted-foreground)",
              lineHeight: 1.4,
              display: "block",
              marginBottom: 12,
            }}
          >
            {t("policies.flowBuilder.versionsSidebar.productionHint")}
          </span>
          <Button
            onClick={onNewVersion}
            disabled={!accessToken || isCreatingVersion}
            style={{ width: "100%", marginBottom: 12 }}
          >
            {t("policies.flowBuilder.versionsSidebar.newVersion")}
          </Button>
          {isLoading ? (
            <div style={{ display: "flex", justifyContent: "center", padding: 16 }}>
              <UiLoadingSpinner className="size-4" />
            </div>
          ) : versions.length === 0 ? (
            <span style={{ fontSize: 13, color: "var(--color-muted-foreground)" }}>
              {t("policies.flowBuilder.versionsSidebar.empty")}
            </span>
          ) : (
            <div className="flex flex-col gap-1">
              {versions.map((v) => {
                const statusStyle = VERSION_STATUS_STYLES[v.version_status ?? "draft"] ?? VERSION_STATUS_STYLES.draft;
                const isActive = v.policy_id === editingPolicyId;
                return (
                  <button
                    key={v.policy_id}
                    type="button"
                    onClick={() => onSelectVersion(v)}
                    style={{
                      width: "100%",
                      textAlign: "left",
                      padding: "10px 12px",
                      borderRadius: 8,
                      border: isActive ? "1px solid var(--color-info)" : "1px solid var(--color-border)",
                      backgroundColor: isActive
                        ? "color-mix(in oklab, var(--color-info) 10%, transparent)"
                        : "var(--color-card)",
                      cursor: "pointer",
                    }}
                  >
                    <div className="flex items-center justify-between" style={{ marginBottom: 4 }}>
                      <span style={{ fontSize: 13, fontWeight: 600, color: "var(--color-foreground)" }}>
                        {t("policies.flowBuilder.versionsSidebar.versionNumber", { number: v.version_number ?? 1 })}
                      </span>
                      <span
                        style={{
                          fontSize: 10,
                          fontWeight: 600,
                          textTransform: "uppercase",
                          backgroundColor: statusStyle.bg,
                          color: statusStyle.color,
                          padding: "2px 6px",
                          borderRadius: 4,
                        }}
                      >
                        {v.version_status ?? "draft"}
                      </span>
                    </div>
                  </button>
                );
              })}
            </div>
          )}

          {/* Publish / Promote to production for selected version */}
          {(canPublish || canPromote) && (
            <div style={{ marginTop: 12, paddingTop: 12, borderTop: "1px solid var(--color-border)" }}>
              {canPublish && (
                <>
                  <Button
                    variant="secondary"
                    onClick={onPublish}
                    disabled={!accessToken || isUpdatingStatus}
                    style={{ width: "100%", marginBottom: 8 }}
                  >
                    {t("policies.flowBuilder.versionsSidebar.publish")}
                  </Button>
                  <span
                    style={{
                      fontSize: 11,
                      color: "var(--color-muted-foreground)",
                      lineHeight: 1.4,
                      display: "block",
                      marginBottom: canPromote ? 8 : 0,
                    }}
                  >
                    {t("policies.flowBuilder.versionsSidebar.publishHint")}
                  </span>
                </>
              )}
              {canPromote && (
                <>
                  <Button
                    onClick={onPromoteToProduction}
                    disabled={!accessToken || isUpdatingStatus}
                    style={{ width: "100%", marginBottom: 8 }}
                  >
                    {t("policies.flowBuilder.versionsSidebar.promote")}
                  </Button>
                  <span
                    style={{
                      fontSize: 11,
                      color: "var(--color-muted-foreground)",
                      lineHeight: 1.4,
                      display: "block",
                    }}
                  >
                    {t("policies.flowBuilder.versionsSidebar.promoteHint")}
                  </span>
                </>
              )}
            </div>
          )}
        </div>

        {/* Silent Mirroring section */}
        <div>
          <div className="flex items-center gap-2" style={{ marginBottom: 8 }}>
            <span
              style={{
                fontSize: 11,
                fontWeight: 700,
                textTransform: "uppercase",
                color: "var(--color-muted-foreground)",
                letterSpacing: "0.06em",
              }}
            >
              {t("policies.flowBuilder.versionsSidebar.silentMirroring.title")}
            </span>
            <span
              style={{
                fontSize: 10,
                fontWeight: 600,
                backgroundColor: "color-mix(in oklab, var(--color-info) 10%, transparent)",
                color: "var(--color-info)",
                padding: "2px 6px",
                borderRadius: 4,
              }}
            >
              {t("policies.flowBuilder.versionsSidebar.silentMirroring.comingSoon")}
            </span>
          </div>
          <span
            style={{
              fontSize: 12,
              color: "var(--color-muted-foreground)",
              lineHeight: 1.5,
              display: "block",
            }}
          >
            {t("policies.flowBuilder.versionsSidebar.silentMirroring.description")}
          </span>
        </div>
      </div>
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
  onVersionCreated?: (newPolicy: Policy) => void;
  onSelectVersion?: (policy: Policy) => void;
  onVersionStatusUpdated?: (updatedPolicy: Policy) => void;
}

export const FlowBuilderPage: React.FC<FlowBuilderPageProps> = ({
  onBack,
  onSuccess,
  accessToken,
  editingPolicy,
  availableGuardrails,
  createPolicy,
  updatePolicy,
  onVersionCreated,
  onSelectVersion,
  onVersionStatusUpdated,
}) => {
  const { t } = useTranslation();
  const isEditing = !!editingPolicy?.policy_id;
  const showVersionsSidebar = !!editingPolicy?.policy_name;

  const [policyName, setPolicyName] = useState(editingPolicy?.policy_name || "");
  const [description, setDescription] = useState(editingPolicy?.description || "");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showTestPanel, setShowTestPanel] = useState(false);
  const [pipeline, setPipeline] = useState<GuardrailPipeline>(() => derivePipelineFromPolicy(editingPolicy));
  const [versions, setVersions] = useState<Policy[]>([]);
  const [isVersionsLoading, setIsVersionsLoading] = useState(false);
  const [isCreatingVersion, setIsCreatingVersion] = useState(false);
  const [isUpdatingStatus, setIsUpdatingStatus] = useState(false);

  // Sync local state when editingPolicy changes (e.g. user switched version)
  React.useEffect(() => {
    setPolicyName(editingPolicy?.policy_name || "");
    setDescription(editingPolicy?.description || "");
    setPipeline(derivePipelineFromPolicy(editingPolicy));
  }, [
    editingPolicy?.policy_id,
    editingPolicy?.policy_name,
    editingPolicy?.description,
    editingPolicy?.pipeline,
    editingPolicy?.guardrails_add,
  ]);

  // Fetch versions when editing an existing policy by name
  React.useEffect(() => {
    if (!showVersionsSidebar || !editingPolicy?.policy_name || !accessToken) {
      setVersions([]);
      return;
    }
    let cancelled = false;
    setIsVersionsLoading(true);
    listPolicyVersions(accessToken, editingPolicy.policy_name)
      .then((res) => {
        if (!cancelled) setVersions(res.versions || []);
      })
      .catch(() => {
        if (!cancelled) setVersions([]);
      })
      .finally(() => {
        if (!cancelled) setIsVersionsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [showVersionsSidebar, editingPolicy?.policy_name, accessToken]);

  const handleNewVersion = async () => {
    if (!accessToken || !editingPolicy?.policy_name) return;
    setIsCreatingVersion(true);
    try {
      const newPolicy = await createPolicyVersion(accessToken, editingPolicy.policy_name);
      toast.success(t("policies.flowBuilder.page.toasts.versionCreated"));
      onVersionCreated?.(newPolicy);
      const list = await listPolicyVersions(accessToken, editingPolicy.policy_name);
      setVersions(list.versions ?? []);
    } catch (error) {
      toast.fromError(
        t("policies.flowBuilder.page.toasts.createVersionFailed", {
          error: error instanceof Error ? error.message : String(error),
        }),
      );
    } finally {
      setIsCreatingVersion(false);
    }
  };

  const handleSelectVersion = (policy: Policy) => {
    onSelectVersion?.(policy);
  };

  const handlePublishVersion = async () => {
    if (!accessToken || !editingPolicy?.policy_id) return;
    setIsUpdatingStatus(true);
    try {
      const updated = await updatePolicyVersionStatus(accessToken, editingPolicy.policy_id, "published");
      toast.success(t("policies.flowBuilder.page.toasts.versionPublished"));
      const list = await listPolicyVersions(accessToken, editingPolicy.policy_name ?? "");
      setVersions(list.versions ?? []);
      onVersionStatusUpdated?.(updated);
    } catch (error) {
      toast.fromError(
        t("policies.flowBuilder.page.toasts.publishFailed", {
          error: error instanceof Error ? error.message : String(error),
        }),
      );
    } finally {
      setIsUpdatingStatus(false);
    }
  };

  const handlePromoteToProduction = async () => {
    if (!accessToken || !editingPolicy?.policy_id) return;
    setIsUpdatingStatus(true);
    try {
      const updated = await updatePolicyVersionStatus(accessToken, editingPolicy.policy_id, "production");
      toast.success(t("policies.flowBuilder.page.toasts.versionPromoted"));
      const list = await listPolicyVersions(accessToken, editingPolicy.policy_name ?? "");
      setVersions(list.versions ?? []);
      onVersionStatusUpdated?.(updated);
    } catch (error) {
      toast.fromError(
        t("policies.flowBuilder.page.toasts.promoteFailed", {
          error: error instanceof Error ? error.message : String(error),
        }),
      );
    } finally {
      setIsUpdatingStatus(false);
    }
  };

  const handleSave = async () => {
    if (!policyName.trim()) {
      toast.error(t("policies.flowBuilder.page.toasts.nameRequired"));
      return;
    }
    if (!accessToken) {
      toast.error(t("policies.flowBuilder.page.toasts.noAccessToken"));
      return;
    }

    const emptySteps = pipeline.steps.filter((s) => !s.guardrail);
    if (emptySteps.length > 0) {
      toast.error(t("policies.flowBuilder.page.toasts.guardrailRequired"));
      return;
    }

    setIsSubmitting(true);
    try {
      const guardrailsFromPipeline = pipeline.steps.map((s) => s.guardrail).filter(Boolean);

      const data: PolicyCreateRequest | PolicyUpdateRequest = {
        policy_name: policyName,
        description: description || undefined,
        guardrails_add: guardrailsFromPipeline,
        guardrails_remove: [],
        pipeline: pipeline,
      };

      if (isEditing && editingPolicy) {
        await updatePolicy(accessToken, editingPolicy.policy_id, data as PolicyUpdateRequest);
        toast.success(t("policies.flowBuilder.page.toasts.policyUpdated"));
        onSuccess();
      } else {
        await createPolicy(accessToken, data as PolicyCreateRequest);
        toast.success(t("policies.flowBuilder.page.toasts.policyCreated"));
        onSuccess();
        onBack();
      }
    } catch (error) {
      console.error("Failed to save policy:", error);
      toast.fromError(
        t("policies.flowBuilder.page.toasts.saveFailed", {
          error: error instanceof Error ? error.message : String(error),
        }),
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="flex h-full min-h-0 w-full flex-1 flex-col overflow-hidden bg-muted">
      {/* Header bar */}
      <div
        style={{
          borderBottom: "1px solid var(--color-border)",
          backgroundColor: "var(--color-card)",
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
            <ArrowLeft style={{ width: 18, height: 18, color: "var(--color-muted-foreground)" }} />
          </button>
          <span style={{ fontSize: 14, color: "var(--color-muted-foreground)" }}>
            {t("policies.flowBuilder.page.breadcrumb")}
          </span>
          <span style={{ fontSize: 14, color: "var(--color-border)" }}>/</span>
          <Input
            placeholder={t("policies.flowBuilder.page.namePlaceholder")}
            value={policyName}
            onChange={(e) => setPolicyName(e.target.value)}
            disabled={isEditing}
            style={{ width: 240 }}
          />
          <span
            style={{
              fontSize: 11,
              fontWeight: 600,
              backgroundColor: "color-mix(in oklab, var(--color-info) 10%, transparent)",
              color: "var(--color-info)",
              padding: "3px 8px",
              borderRadius: 4,
              letterSpacing: "0.02em",
            }}
          >
            {t("policies.flowBuilder.page.flowBadge")}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="secondary" onClick={onBack}>
            {t("policies.flowBuilder.page.cancel")}
          </Button>
          <Button variant="secondary" onClick={() => setShowTestPanel(!showTestPanel)}>
            {showTestPanel ? t("policies.flowBuilder.page.hideTest") : t("policies.flowBuilder.page.showTest")}
          </Button>
          <Button onClick={handleSave} disabled={isSubmitting}>
            {isEditing ? t("policies.flowBuilder.page.update") : t("policies.flowBuilder.page.save")}
          </Button>
        </div>
      </div>

      {/* Description bar */}
      <div
        style={{
          padding: "8px 24px",
          backgroundColor: "var(--color-card)",
          borderBottom: "1px solid var(--color-border)",
          flexShrink: 0,
        }}
      >
        <Input
          placeholder={t("policies.flowBuilder.page.descriptionPlaceholder")}
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          style={{ maxWidth: 500 }}
        />
      </div>

      {/* Sidebar (when editing) + Flow builder canvas + test panel */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
        {showVersionsSidebar && (
          <PolicyVersionsSidebar
            policyName={policyName}
            editingPolicyId={editingPolicy?.policy_id ?? null}
            editingVersionStatus={editingPolicy?.version_status}
            accessToken={accessToken}
            versions={versions}
            isLoading={isVersionsLoading}
            isCreatingVersion={isCreatingVersion}
            isUpdatingStatus={isUpdatingStatus}
            onNewVersion={handleNewVersion}
            onSelectVersion={handleSelectVersion}
            onPublish={handlePublishVersion}
            onPromoteToProduction={handlePromoteToProduction}
          />
        )}
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
            <PipelineFlowBuilder pipeline={pipeline} onChange={setPipeline} availableGuardrails={availableGuardrails} />
          </div>
        </div>

        {showTestPanel && (
          <PipelineTestPanel pipeline={pipeline} accessToken={accessToken} onClose={() => setShowTestPanel(false)} />
        )}
      </div>
    </div>
  );
};

export { createDefaultStep };
export default PipelineFlowBuilder;
