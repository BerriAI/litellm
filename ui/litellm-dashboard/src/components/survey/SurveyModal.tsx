import React, { useState } from "react";
import { X, MessageSquare, ArrowRight, ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Checkbox } from "@/components/ui/checkbox";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";

interface SurveyModalProps {
  isOpen: boolean;
  onClose: () => void;
  onComplete: () => void;
}

const REASONS_OPTIONS = [
  {
    id: "oss_adoption",
    label: "OSS Adoption",
    description: "Stars, contributors, forks, community support",
  },
  {
    id: "ai_integration",
    label: "AI Integration",
    description: "LiteLLM had the logging/guardrail integration we needed - Langfuse, OTEL, S3 logging, Azure Content Safety guardrails",
  },
  {
    id: "unified_api",
    label: "Unified API",
    description: "LiteLLM had the best OpenAI-compatible API across providers - OpenAI, Anthropic, Gemini, etc.",
  },
  {
    id: "breadth_of_models",
    label: "Breadth of Models/Providers",
    description: "LiteLLM had the provider + endpoint combinations we needed - /ocr endpoint with Mistral OCR, /batches endppint with Bedrock API, etc.",
  },
  {
    id: "other",
    label: "Other",
    description: "Something else not listed above",
  },
];

type SurveyData = {
  usingAtCompany: boolean | null;
  companyName: string;
  startDate: string;
  reasons: string[];
  otherReason: string;
  email: string;
};

export function SurveyModal({ isOpen, onClose, onComplete }: SurveyModalProps) {
  const [step, setStep] = useState(1);
  const [data, setData] = useState<SurveyData>({
    usingAtCompany: null,
    companyName: "",
    startDate: "",
    reasons: [],
    otherReason: "",
    email: "",
  });
  const [isSubmitting, setIsSubmitting] = useState(false);

  const totalSteps = data.usingAtCompany === true ? 5 : 4;

  if (!isOpen) return null;

  const handleNext = () => {
    if (step === 1 && data.usingAtCompany === false) {
      setStep(3);
    } else if (step < 5) {
      setStep(step + 1);
    } else {
      handleSubmit();
    }
  };

  const handleBack = () => {
    if (step === 3 && data.usingAtCompany === false) {
      setStep(1);
    } else {
      setStep(step - 1);
    }
  };

  const handleSubmit = async () => {
    setIsSubmitting(true);
    try {
      const reasonLabels: Record<string, string> = {
        oss_adoption: "OSS Adoption (stars, contributors, forks)",
        ai_integration: "AI Integration (Langfuse, OTEL, S3, Azure Content Safety)",
        unified_api: "Unified API (OpenAI-compatible)",
        breadth_of_models: "Breadth of Models/Providers (/ocr, /batches, Bedrock, Azure OCR)",
      };

      const readableReasons = data.reasons.map((r) => {
        if (r === "other" && data.otherReason) {
          return `Other: ${data.otherReason}`;
        }
        return reasonLabels[r] || r;
      });

      const feedbackUrl = "https://feedback.litellm.ai/survey";

      const formData = new URLSearchParams({
        "entry.2015264290": data.usingAtCompany ? "Yes" : "No",
        "entry.1876243786": data.companyName || "",
        "entry.1282591459": data.startDate,
        "entry.393456108": readableReasons.join(", "),
        "entry.928142208": data.email || "",
      });

      await fetch(feedbackUrl, {
        method: "POST",
        mode: "no-cors",
        body: formData,
      });
    } catch (error) {
      console.error("Failed to submit survey:", error);
    }
    setIsSubmitting(false);
    onComplete();
  };

  const updateData = (key: keyof SurveyData, value: boolean | string | string[] | null) => {
    setData((prev) => ({
      ...prev,
      [key]: value,
    }));
  };

  const toggleReason = (reasonId: string) => {
    setData((prev) => ({
      ...prev,
      reasons: prev.reasons.includes(reasonId)
        ? prev.reasons.filter((r) => r !== reasonId)
        : [...prev.reasons, reasonId],
    }));
  };

  const isStepValid = () => {
    if (step === 1) return data.usingAtCompany !== null;
    if (step === 2) return data.companyName.trim().length > 0;
    if (step === 3) return data.startDate !== "";
    if (step === 4) {
      if (data.reasons.includes("other")) {
        return data.reasons.length > 0 && data.otherReason.trim().length > 0;
      }
      return data.reasons.length > 0;
    }
    if (step === 5) return true;
    return false;
  };

  const getStepNumber = () => {
    if (data.usingAtCompany === false) {
      if (step === 1) return 1;
      if (step === 3) return 2;
      if (step === 4) return 3;
      if (step === 5) return 4;
    }
    return step;
  };

  const renderStepContent = () => {
    if (step === 1) {
      return (
        <div className="space-y-6">
          <h2 className="text-2xl font-bold text-gray-900">Are you using LiteLLM at your company?</h2>
          <p className="text-gray-500">Help us understand how our product is being used in professional environments.</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-4">
            <button
              onClick={() => updateData("usingAtCompany", true)}
              className={`p-6 rounded-lg border-2 text-left transition-all ${
                data.usingAtCompany === true
                  ? "border-blue-600 bg-blue-50 ring-1 ring-blue-600"
                  : "border-gray-200 hover:border-blue-300 hover:bg-gray-50"
              }`}
            >
              <span className="block text-lg font-semibold text-gray-900 mb-1">Yes</span>
              <span className="text-sm text-gray-500">We use it for work</span>
            </button>
            <button
              onClick={() => updateData("usingAtCompany", false)}
              className={`p-6 rounded-lg border-2 text-left transition-all ${
                data.usingAtCompany === false
                  ? "border-blue-600 bg-blue-50 ring-1 ring-blue-600"
                  : "border-gray-200 hover:border-blue-300 hover:bg-gray-50"
              }`}
            >
              <span className="block text-lg font-semibold text-gray-900 mb-1">No</span>
              <span className="text-sm text-gray-500">Personal project / Hobby</span>
            </button>
          </div>
        </div>
      );
    }

    if (step === 2 && data.usingAtCompany === true) {
      return (
        <div className="space-y-6">
          <h2 className="text-2xl font-bold text-gray-900">What company are you using LiteLLM at?</h2>
          <p className="text-gray-500">This helps us understand our user base better.</p>
          <Input
            className="h-11 text-base"
            placeholder="Enter your company name"
            value={data.companyName}
            onChange={(e) => updateData("companyName", e.target.value)}
            autoFocus
          />
        </div>
      );
    }

    if (step === 3) {
      return (
        <div className="space-y-6">
          <h2 className="text-2xl font-bold text-gray-900">When did you start using LiteLLM?</h2>
          <RadioGroup
            value={data.startDate}
            onValueChange={(v) => updateData("startDate", v)}
            className="w-full flex flex-col gap-2"
          >
            {["Less than a month ago", "1-3 months ago", "3-6 months ago", "More than 6 months ago"].map((option) => (
              <Label
                key={option}
                htmlFor={`start-${option}`}
                className={`flex items-center p-4 rounded-lg border cursor-pointer transition-all w-full ${
                  data.startDate === option
                    ? "border-blue-600 bg-blue-50 ring-1 ring-blue-600"
                    : "border-gray-200 hover:bg-gray-50"
                }`}
              >
                <RadioGroupItem value={option} id={`start-${option}`} className="mr-2" />
                <span>{option}</span>
              </Label>
            ))}
          </RadioGroup>
        </div>
      );
    }

    if (step === 4) {
      return (
        <div className="space-y-6">
          <h2 className="text-2xl font-bold text-gray-900">Why did you pick LiteLLM over other AI Gateways?</h2>
          <p className="text-gray-500">Select all that apply.</p>
          <div className="space-y-3">
            {REASONS_OPTIONS.map((option) => {
              const isSelected = data.reasons.includes(option.id);
              return (
                <div key={option.id}>
                  <button
                    type="button"
                    onClick={() => toggleReason(option.id)}
                    className={`w-full flex items-start p-4 rounded-lg border cursor-pointer transition-all text-left ${
                      isSelected
                        ? "border-blue-600 bg-blue-50 ring-1 ring-blue-600"
                        : "border-gray-200 hover:bg-gray-50"
                    }`}
                  >
                    <Checkbox checked={isSelected} className="mt-0.5 pointer-events-none" />
                    <div className="ml-3">
                      <span className="block font-medium text-gray-900">{option.label}</span>
                      <span className="text-sm text-gray-500">{option.description}</span>
                    </div>
                  </button>
                  {option.id === "other" && isSelected && (
                    <Input
                      className="mt-2 ml-7"
                      placeholder="Please specify..."
                      value={data.otherReason}
                      onChange={(e) => updateData("otherReason", e.target.value)}
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

    if (step === 5) {
      return (
        <div className="space-y-6">
          <h2 className="text-2xl font-bold text-gray-900">Want to share more?</h2>
          <p className="text-gray-500">
            Leave your email and we may reach out to learn more about your experience. This is completely optional.
          </p>
          <Input
            className="h-11 text-base"
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

  const isLastStep = step === 5;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6">
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/40 backdrop-blur-sm" onClick={onClose} />

      {/* Modal */}
      <div className="relative w-full max-w-lg bg-white rounded-xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh] transform transition-all duration-300 ease-out">
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-100 flex items-center justify-between bg-gray-50/50">
          <div className="flex items-center gap-2 text-blue-600">
            <MessageSquare className="h-5 w-5" />
            <span className="font-semibold text-sm tracking-wide uppercase">Quick Feedback</span>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors p-1 rounded-full hover:bg-gray-100"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Progress Bar */}
        <Progress value={(getStepNumber() / totalSteps) * 100} className="h-1 rounded-none" />

        {/* Content */}
        <div className="p-8 flex-1 overflow-y-auto">{renderStepContent()}</div>

        {/* Footer */}
        <div className="px-6 py-4 bg-gray-50 border-t border-gray-200 flex items-center justify-between">
          <div className="text-sm text-gray-500 font-medium">
            Step {getStepNumber()} of {totalSteps}
          </div>
          <div className="flex gap-3">
            {step > 1 && (
              <Button variant="outline" onClick={handleBack} disabled={isSubmitting}>
                <ArrowLeft className="h-4 w-4 mr-1" />
                Back
              </Button>
            )}
            <Button
              onClick={handleNext}
              disabled={!isStepValid() || isSubmitting}
              className="min-w-[100px]"
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
