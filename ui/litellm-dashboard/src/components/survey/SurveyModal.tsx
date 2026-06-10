import React, { useState, useMemo } from "react";
import { X, MessageSquare, ArrowRight, ArrowLeft } from "lucide-react";
import { Button, Input, Radio, Space, Progress, Checkbox } from "antd";
import { useTranslation } from "react-i18next";
import { TFunction } from "i18next";

interface SurveyModalProps {
  isOpen: boolean;
  onClose: () => void;
  onComplete: () => void;
}

type ReasonOption = {
  id: string;
  label: string;
  description: string;
};

const getReasonsOptions = (t: TFunction): ReasonOption[] => [
  {
    id: "oss_adoption",
    label: t("survey.surveyModal.reasonOssLabel"),
    description: t("survey.surveyModal.reasonOssDesc"),
  },
  {
    id: "ai_integration",
    label: t("survey.surveyModal.reasonAiIntegrationLabel"),
    description: t("survey.surveyModal.reasonAiIntegrationDesc"),
  },
  {
    id: "unified_api",
    label: t("survey.surveyModal.reasonUnifiedApiLabel"),
    description: t("survey.surveyModal.reasonUnifiedApiDesc"),
  },
  {
    id: "breadth_of_models",
    label: t("survey.surveyModal.reasonBreadthLabel"),
    description: t("survey.surveyModal.reasonBreadthDesc"),
  },
  {
    id: "other",
    label: t("survey.surveyModal.reasonOtherLabel"),
    description: t("survey.surveyModal.reasonOtherDesc"),
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
  const { t } = useTranslation();
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

  const reasonsOptions = useMemo(() => getReasonsOptions(t), [t]);

  // Steps: 1=company?, 2=company name (conditional), 3=when, 4=why, 5=email
  // If not at company: skip step 2, so total is 4
  // If at company: total is 5
  const totalSteps = data.usingAtCompany === true ? 5 : 4;

  if (!isOpen) return null;

  const handleNext = () => {
    // Skip company name step if not using at company
    if (step === 1 && data.usingAtCompany === false) {
      setStep(3); // Skip to "when did you start"
    } else if (step < 5) {
      setStep(step + 1);
    } else {
      handleSubmit();
    }
  };

  const handleBack = () => {
    if (step === 3 && data.usingAtCompany === false) {
      setStep(1); // Go back to first question if we skipped company name
    } else {
      setStep(step - 1);
    }
  };

  const handleSubmit = async () => {
    setIsSubmitting(true);
    try {
      // Map reason IDs to readable labels
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

      // Submit to feedback endpoint (redirects to Google Form)
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
      // Silently fail - don't block the user experience
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
      // If "other" is selected, require the text field
      if (data.reasons.includes("other")) {
        return data.reasons.length > 0 && data.otherReason.trim().length > 0;
      }
      return data.reasons.length > 0;
    }
    if (step === 5) return true; // Email is optional
    return false;
  };

  const getStepNumber = () => {
    if (data.usingAtCompany === false) {
      // When not at company: skip step 2, so steps 3,4,5 become 2,3,4
      if (step === 1) return 1;
      if (step === 3) return 2;
      if (step === 4) return 3;
      if (step === 5) return 4;
    }
    return step;
  };

  const startDateOptions = [
    { value: "Less than a month ago", label: t("survey.surveyModal.lessThanMonth") },
    { value: "1-3 months ago", label: t("survey.surveyModal.oneToThreeMonths") },
    { value: "3-6 months ago", label: t("survey.surveyModal.threeToSixMonths") },
    { value: "More than 6 months ago", label: t("survey.surveyModal.moreThanSixMonths") },
  ];

  const renderStepContent = () => {
    // Step 1: Using at company?
    if (step === 1) {
      return (
        <div className="space-y-6">
          <h2 className="text-2xl font-bold text-gray-900">{t("survey.surveyModal.step1Title")}</h2>
          <p className="text-gray-500">{t("survey.surveyModal.step1Description")}</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-4">
            <button
              onClick={() => updateData("usingAtCompany", true)}
              className={`p-6 rounded-lg border-2 text-left transition-all ${
                data.usingAtCompany === true
                  ? "border-blue-600 bg-blue-50 ring-1 ring-blue-600"
                  : "border-gray-200 hover:border-blue-300 hover:bg-gray-50"
              }`}
            >
              <span className="block text-lg font-semibold text-gray-900 mb-1">{t("common.yes")}</span>
              <span className="text-sm text-gray-500">{t("survey.surveyModal.yesSubLabel")}</span>
            </button>
            <button
              onClick={() => updateData("usingAtCompany", false)}
              className={`p-6 rounded-lg border-2 text-left transition-all ${
                data.usingAtCompany === false
                  ? "border-blue-600 bg-blue-50 ring-1 ring-blue-600"
                  : "border-gray-200 hover:border-blue-300 hover:bg-gray-50"
              }`}
            >
              <span className="block text-lg font-semibold text-gray-900 mb-1">{t("common.no")}</span>
              <span className="text-sm text-gray-500">{t("survey.surveyModal.noSubLabel")}</span>
            </button>
          </div>
        </div>
      );
    }

    // Step 2: Company name (only if using at company)
    if (step === 2 && data.usingAtCompany === true) {
      return (
        <div className="space-y-6">
          <h2 className="text-2xl font-bold text-gray-900">{t("survey.surveyModal.step2Title")}</h2>
          <p className="text-gray-500">{t("survey.surveyModal.step2Description")}</p>
          <Input
            size="large"
            placeholder={t("survey.surveyModal.companyPlaceholder")}
            value={data.companyName}
            onChange={(e) => updateData("companyName", e.target.value)}
            autoFocus
          />
        </div>
      );
    }

    // Step 3: When did you start?
    if (step === 3) {
      return (
        <div className="space-y-6">
          <h2 className="text-2xl font-bold text-gray-900">{t("survey.surveyModal.step3Title")}</h2>
          <Radio.Group
            value={data.startDate}
            onChange={(e) => updateData("startDate", e.target.value)}
            className="w-full"
          >
            <Space direction="vertical" className="w-full">
              {startDateOptions.map((option) => (
                <label
                  key={option.value}
                  className={`flex items-center p-4 rounded-lg border cursor-pointer transition-all w-full ${
                    data.startDate === option.value
                      ? "border-blue-600 bg-blue-50 ring-1 ring-blue-600"
                      : "border-gray-200 hover:bg-gray-50"
                  }`}
                >
                  <Radio value={option.value}>{option.label}</Radio>
                </label>
              ))}
            </Space>
          </Radio.Group>
        </div>
      );
    }

    // Step 4: Why did you pick LiteLLM?
    if (step === 4) {
      return (
        <div className="space-y-6">
          <h2 className="text-2xl font-bold text-gray-900">{t("survey.surveyModal.step4Title")}</h2>
          <p className="text-gray-500">{t("survey.surveyModal.step4Description")}</p>
          <div className="space-y-3">
            {reasonsOptions.map((option) => {
              const isSelected = data.reasons.includes(option.id);
              return (
                <div key={option.id}>
                  <div
                    role="button"
                    tabIndex={0}
                    onClick={() => toggleReason(option.id)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        toggleReason(option.id);
                      }
                    }}
                    className={`flex items-start p-4 rounded-lg border cursor-pointer transition-all ${
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
                  </div>
                  {/* Show text input if "Other" is selected */}
                  {option.id === "other" && isSelected && (
                    <Input
                      className="mt-2 ml-7"
                      placeholder={t("survey.surveyModal.otherPlaceholder")}
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

    // Step 5: Email (optional)
    if (step === 5) {
      return (
        <div className="space-y-6">
          <h2 className="text-2xl font-bold text-gray-900">{t("survey.surveyModal.step5Title")}</h2>
          <p className="text-gray-500">{t("survey.surveyModal.step5Description")}</p>
          <Input
            size="large"
            type="email"
            placeholder={t("survey.surveyModal.emailPlaceholder")}
            value={data.email}
            onChange={(e) => updateData("email", e.target.value)}
            autoFocus
          />
          <p className="text-xs text-gray-400">{t("survey.surveyModal.emailNote")}</p>
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
            <span className="font-semibold text-sm tracking-wide uppercase">{t("survey.surveyModal.headerLabel")}</span>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors p-1 rounded-full hover:bg-gray-100"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Progress Bar */}
        <Progress
          percent={(getStepNumber() / totalSteps) * 100}
          showInfo={false}
          strokeColor="#2563eb"
          className="m-0"
        />

        {/* Content */}
        <div className="p-8 flex-1 overflow-y-auto">{renderStepContent()}</div>

        {/* Footer */}
        <div className="px-6 py-4 bg-gray-50 border-t border-gray-200 flex items-center justify-between">
          <div className="text-sm text-gray-500 font-medium">
            {t("survey.surveyModal.stepIndicator", { current: getStepNumber(), total: totalSteps })}
          </div>
          <div className="flex gap-3">
            {step > 1 && (
              <Button onClick={handleBack} disabled={isSubmitting} icon={<ArrowLeft className="h-4 w-4" />}>
                {t("common.back")}
              </Button>
            )}
            <Button
              type="primary"
              onClick={handleNext}
              disabled={!isStepValid() || isSubmitting}
              loading={isSubmitting}
              className="min-w-[100px]"
            >
              {isLastStep ? t("common.submit") : t("common.next")}
              {!isLastStep && <ArrowRight className="ml-2 h-4 w-4" />}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
