import React, { useState } from "react";
import { X, Code, ArrowRight, ArrowLeft } from "lucide-react";
import { Button, Input, Radio, Space, Progress, Checkbox } from "antd";

interface ClaudeCodeModalProps {
  isOpen: boolean;
  onClose: () => void;
  onComplete: () => void;
}

const FEATURE_OPTIONS = [
  {
    id: "model_compatibility",
    label: "Model Compatibility",
    description: "Works seamlessly with Claude models through LiteLLM",
  },
  {
    id: "performance",
    label: "Performance & Speed",
    description: "Fast response times and reliable connections",
  },
  {
    id: "debugging",
    label: "Debugging & Observability",
    description: "Easy to debug and monitor API calls",
  },
  {
    id: "cost_tracking",
    label: "Cost Tracking",
    description: "Helpful for tracking usage and costs",
  },
  {
    id: "other",
    label: "Other",
    description: "Something else not listed above",
  },
];

const IMPROVEMENT_OPTIONS = [
  {
    id: "documentation",
    label: "Documentation",
    description: "Better guides for Claude Code integration",
  },
  {
    id: "error_messages",
    label: "Error Messages",
    description: "Clearer error messages and debugging info",
  },
  {
    id: "performance_optimization",
    label: "Performance",
    description: "Faster response times or lower latency",
  },
  {
    id: "additional_features",
    label: "Additional Features",
    description: "More features specific to Claude Code workflows",
  },
  {
    id: "nothing",
    label: "Nothing - It's great!",
    description: "I'm satisfied with the current experience",
  },
  {
    id: "other",
    label: "Other",
    description: "Something else not listed above",
  },
];

type SurveyData = {
  satisfaction: string;
  features: string[];
  improvements: string[];
  improvementOther: string;
  email: string;
};

export function ClaudeCodeModal({ isOpen, onClose, onComplete }: ClaudeCodeModalProps) {
  const [step, setStep] = useState(1);
  const [data, setData] = useState<SurveyData>({
    satisfaction: "",
    features: [],
    improvements: [],
    improvementOther: "",
    email: "",
  });
  const [isSubmitting, setIsSubmitting] = useState(false);

  const totalSteps = 4;

  if (!isOpen) return null;

  const handleNext = () => {
    if (step < 4) {
      setStep(step + 1);
    } else {
      handleSubmit();
    }
  };

  const handleBack = () => {
    setStep(step - 1);
  };

  const handleSubmit = async () => {
    setIsSubmitting(true);
    try {
      // Map IDs to readable labels
      const featureLabels: Record<string, string> = {
        model_compatibility: "Model Compatibility",
        performance: "Performance & Speed",
        debugging: "Debugging & Observability",
        cost_tracking: "Cost Tracking",
      };

      const improvementLabels: Record<string, string> = {
        documentation: "Documentation",
        error_messages: "Error Messages",
        performance_optimization: "Performance",
        additional_features: "Additional Features",
        nothing: "Nothing - It's great!",
      };

      const readableFeatures = data.features.map((f) => {
        if (f === "other") return "Other";
        return featureLabels[f] || f;
      });

      const readableImprovements = data.improvements.map((i) => {
        if (i === "other" && data.improvementOther) {
          return `Other: ${data.improvementOther}`;
        }
        return improvementLabels[i] || i;
      });

      await fetch("https://hooks.zapier.com/hooks/catch/16331268/ugms6w0/", {
        method: "POST",
        mode: "no-cors",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          surveyType: "Claude Code Feedback",
          satisfaction: data.satisfaction,
          features: readableFeatures.join(", "),
          improvements: readableImprovements.join(", "),
          improvementOther: data.improvementOther || null,
          email: data.email || null,
          submittedAt: new Date().toISOString(),
        }),
      });
    } catch (error) {
      // Silently fail - don't block the user experience
      console.error("Failed to submit Claude Code feedback:", error);
    }
    setIsSubmitting(false);
    onComplete();
  };

  const updateData = (key: keyof SurveyData, value: string | string[]) => {
    setData((prev) => ({
      ...prev,
      [key]: value,
    }));
  };

  const toggleFeature = (featureId: string) => {
    setData((prev) => ({
      ...prev,
      features: prev.features.includes(featureId)
        ? prev.features.filter((f) => f !== featureId)
        : [...prev.features, featureId],
    }));
  };

  const toggleImprovement = (improvementId: string) => {
    setData((prev) => ({
      ...prev,
      improvements: prev.improvements.includes(improvementId)
        ? prev.improvements.filter((i) => i !== improvementId)
        : [...prev.improvements, improvementId],
    }));
  };

  const isStepValid = () => {
    if (step === 1) return data.satisfaction !== "";
    if (step === 2) return data.features.length > 0;
    if (step === 3) {
      // If "other" is selected, require the text field
      if (data.improvements.includes("other")) {
        return data.improvements.length > 0 && data.improvementOther.trim().length > 0;
      }
      return data.improvements.length > 0;
    }
    if (step === 4) return true; // Email is optional
    return false;
  };

  const renderStepContent = () => {
    // Step 1: Satisfaction rating
    if (step === 1) {
      return (
        <div className="space-y-6">
          <h2 className="text-2xl font-bold text-gray-900">How satisfied are you with using LiteLLM in Claude Code?</h2>
          <Radio.Group
            value={data.satisfaction}
            onChange={(e) => updateData("satisfaction", e.target.value)}
            className="w-full"
          >
            <Space direction="vertical" className="w-full">
              {[
                "Very Satisfied",
                "Satisfied",
                "Neutral",
                "Dissatisfied",
                "Very Dissatisfied",
              ].map((option) => (
                <label
                  key={option}
                  className={`flex items-center p-4 rounded-lg border cursor-pointer transition-all w-full ${
                    data.satisfaction === option
                      ? "border-purple-600 bg-purple-50 ring-1 ring-purple-600"
                      : "border-gray-200 hover:bg-gray-50"
                  }`}
                >
                  <Radio value={option}>{option}</Radio>
                </label>
              ))}
            </Space>
          </Radio.Group>
        </div>
      );
    }

    // Step 2: What features do you value?
    if (step === 2) {
      return (
        <div className="space-y-6">
          <h2 className="text-2xl font-bold text-gray-900">What do you value most about using LiteLLM with Claude Code?</h2>
          <p className="text-gray-500">Select all that apply.</p>
          <div className="space-y-3">
            {FEATURE_OPTIONS.map((option) => {
              const isSelected = data.features.includes(option.id);
              return (
                <div
                  key={option.id}
                  role="button"
                  tabIndex={0}
                  onClick={() => toggleFeature(option.id)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      toggleFeature(option.id);
                    }
                  }}
                  className={`flex items-start p-4 rounded-lg border cursor-pointer transition-all ${
                    isSelected
                      ? "border-purple-600 bg-purple-50 ring-1 ring-purple-600"
                      : "border-gray-200 hover:bg-gray-50"
                  }`}
                >
                  <Checkbox checked={isSelected} className="mt-0.5 pointer-events-none" />
                  <div className="ml-3">
                    <span className="block font-medium text-gray-900">{option.label}</span>
                    <span className="text-sm text-gray-500">{option.description}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      );
    }

    // Step 3: What could be improved?
    if (step === 3) {
      return (
        <div className="space-y-6">
          <h2 className="text-2xl font-bold text-gray-900">What could we improve for Claude Code users?</h2>
          <p className="text-gray-500">Select all that apply.</p>
          <div className="space-y-3">
            {IMPROVEMENT_OPTIONS.map((option) => {
              const isSelected = data.improvements.includes(option.id);
              return (
                <div key={option.id}>
                  <div
                    role="button"
                    tabIndex={0}
                    onClick={() => toggleImprovement(option.id)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        toggleImprovement(option.id);
                      }
                    }}
                    className={`flex items-start p-4 rounded-lg border cursor-pointer transition-all ${
                      isSelected
                        ? "border-purple-600 bg-purple-50 ring-1 ring-purple-600"
                        : "border-gray-200 hover:bg-gray-50"
                    }`}
                  >
                    <Checkbox checked={isSelected} className="mt-0.5 pointer-events-none" />
                    <div className="ml-3">
                      <span className="block font-medium text-gray-900">{option.label}</span>
                      <span className="text-sm text-gray-500">{option.description}</span>
                    </div>
                  </div>
                  {/* Show text input if "Other" is selected */}
                  {option.id === "other" && isSelected && (
                    <Input
                      className="mt-2 ml-7"
                      placeholder="Please specify..."
                      value={data.improvementOther}
                      onChange={(e) => updateData("improvementOther", e.target.value)}
                      onClick={(e) => e.stopPropagation()}
                      autoFocus
                    />
                  )}
                </div>
              );
            })}
          </div>
        </div>
      );
    }

    // Step 4: Email (optional)
    if (step === 4) {
      return (
        <div className="space-y-6">
          <h2 className="text-2xl font-bold text-gray-900">Want to discuss further?</h2>
          <p className="text-gray-500">
            Leave your email and we may reach out to learn more about your Claude Code experience. This is completely optional.
          </p>
          <Input
            size="large"
            type="email"
            placeholder="your@email.com (optional)"
            value={data.email}
            onChange={(e) => updateData("email", e.target.value)}
            autoFocus
          />
          <p className="text-xs text-gray-400">
            We will only use this to follow up on your feedback. No spam, ever.
          </p>
        </div>
      );
    }

    return null;
  };

  const isLastStep = step === 4;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6">
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/40 backdrop-blur-sm" onClick={onClose} />

      {/* Modal */}
      <div className="relative w-full max-w-lg bg-white rounded-xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh] transform transition-all duration-300 ease-out">
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-100 flex items-center justify-between bg-gray-50/50">
          <div className="flex items-center gap-2 text-purple-600">
            <Code className="h-5 w-5" />
            <span className="font-semibold text-sm tracking-wide uppercase">Claude Code Feedback</span>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors p-1 rounded-full hover:bg-gray-100"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Progress Bar */}
        <Progress percent={(step / totalSteps) * 100} showInfo={false} strokeColor="#7c3aed" className="m-0" />

        {/* Content */}
        <div className="p-8 flex-1 overflow-y-auto">{renderStepContent()}</div>

        {/* Footer */}
        <div className="px-6 py-4 bg-gray-50 border-t border-gray-200 flex items-center justify-between">
          <div className="text-sm text-gray-500 font-medium">
            Step {step} of {totalSteps}
          </div>
          <div className="flex gap-3">
            {step > 1 && (
              <Button onClick={handleBack} disabled={isSubmitting} icon={<ArrowLeft className="h-4 w-4" />}>
                Back
              </Button>
            )}
            <Button
              type="primary"
              onClick={handleNext}
              disabled={!isStepValid() || isSubmitting}
              loading={isSubmitting}
              className="min-w-[100px]"
              style={{ backgroundColor: '#7c3aed', borderColor: '#7c3aed' }}
            >
              {isLastStep ? "Submit" : "Next"}
              {!isLastStep && <ArrowRight className="ml-2 h-4 w-4" />}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

