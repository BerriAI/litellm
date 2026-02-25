import React, { useState, useEffect } from "react";
import { Modal, Spin, Radio, Select } from "antd";
import { Button, TextInput } from "@tremor/react";
import { modelHubCall, enrichPolicyTemplateStream } from "../networking";

interface TemplateParameter {
  name: string;
  label: string;
  type: string;
  required: boolean;
  placeholder?: string;
}

interface TemplateParameterModalProps {
  visible: boolean;
  template: any;
  onConfirm: (
    parameters: Record<string, string>,
    enrichmentOptions?: { model?: string; competitors?: string[] }
  ) => void;
  onCancel: () => void;
  isLoading?: boolean;
  accessToken: string;
}

const TemplateParameterModal: React.FC<TemplateParameterModalProps> = ({
  visible,
  template,
  onConfirm,
  onCancel,
  isLoading = false,
  accessToken,
}) => {
  const [parameterValues, setParameterValues] = useState<Record<string, string>>({});
  const [competitorMode, setCompetitorMode] = useState<"ai" | "manual">("ai");
  const [selectedModel, setSelectedModel] = useState<string | undefined>(undefined);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [competitorTags, setCompetitorTags] = useState<string[]>([]);
  const [variationsMap, setVariationsMap] = useState<Record<string, string[]>>({});
  const [isGenerating, setIsGenerating] = useState(false);
  const [refinementInput, setRefinementInput] = useState("");
  const [isRefining, setIsRefining] = useState(false);
  const [hasGenerated, setHasGenerated] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");

  const parameters: TemplateParameter[] = template?.parameters || [];
  const hasEnrichment = !!template?.llm_enrichment;
  const enrichmentParam = hasEnrichment ? template.llm_enrichment.parameter : null;

  const nonEnrichmentParams = hasEnrichment
    ? parameters.filter((p) => p.name !== enrichmentParam)
    : parameters;

  useEffect(() => {
    if (visible && template) {
      const initial: Record<string, string> = {};
      parameters.forEach((p) => {
        initial[p.name] = "";
      });
      setParameterValues(initial);
      setCompetitorMode("ai");
      setSelectedModel(undefined);
      setCompetitorTags([]);
      setVariationsMap({});
      setIsGenerating(false);
      setRefinementInput("");
      setIsRefining(false);
      setHasGenerated(false);
      setStatusMessage("");
    }
  }, [visible, template]);

  useEffect(() => {
    if (visible && hasEnrichment && competitorMode === "ai" && availableModels.length === 0) {
      loadModels();
    }
  }, [visible, hasEnrichment, competitorMode]);

  const loadModels = async () => {
    if (!accessToken) return;
    setIsLoadingModels(true);
    try {
      const fetchedModels = await modelHubCall(accessToken);
      if (fetchedModels?.data?.length > 0) {
        const models = fetchedModels.data
          .map((item: any) => item.model_group as string)
          .sort();
        setAvailableModels(models);
      }
    } catch (error) {
      console.error("Error fetching models:", error);
    } finally {
      setIsLoadingModels(false);
    }
  };

  const handleGenerateNames = async () => {
    if (!accessToken || !selectedModel || !template) return;
    const brandName = (parameterValues[enrichmentParam || "brand_name"] || "").trim();
    if (!brandName) return;

    setIsGenerating(true);
    setCompetitorTags([]);
    setVariationsMap({});
    setStatusMessage("");
    try {
      await enrichPolicyTemplateStream(
        accessToken,
        template.id,
        parameterValues,
        selectedModel,
        (name) => {
          setCompetitorTags((prev) => [...prev, name]);
        },
        (result) => {
          setCompetitorTags(result.competitors);
          setVariationsMap(result.competitor_variations || {});
          setIsGenerating(false);
          setHasGenerated(true);
          setStatusMessage("");
        },
        (error) => {
          console.error("Streaming error:", error);
          setIsGenerating(false);
          setStatusMessage("");
        },
        undefined,
        (status) => setStatusMessage(status),
      );
    } catch (error) {
      console.error("Error generating competitor names:", error);
      setIsGenerating(false);
    }
  };

  const handleRefine = async () => {
    if (!accessToken || !selectedModel || !template || !refinementInput.trim()) return;

    setIsRefining(true);
    setStatusMessage("");
    try {
      await enrichPolicyTemplateStream(
        accessToken,
        template.id,
        parameterValues,
        selectedModel,
        (name) => {
          setCompetitorTags((prev) => {
            if (prev.some((t) => t.toLowerCase() === name.toLowerCase())) return prev;
            return [...prev, name];
          });
        },
        (result) => {
          setCompetitorTags(result.competitors);
          setVariationsMap(result.competitor_variations || {});
          setIsRefining(false);
          setRefinementInput("");
          setStatusMessage("");
        },
        (error) => {
          console.error("Refinement error:", error);
          setIsRefining(false);
          setStatusMessage("");
        },
        {
          instruction: refinementInput.trim(),
          existingCompetitors: competitorTags,
        },
        (status) => setStatusMessage(status),
      );
    } catch (error) {
      console.error("Error refining competitor names:", error);
      setIsRefining(false);
    }
  };

  const allNonEnrichmentFilled = nonEnrichmentParams
    .filter((p) => p.required)
    .every((p) => (parameterValues[p.name] || "").trim().length > 0);

  const brandNameFilled = enrichmentParam
    ? (parameterValues[enrichmentParam] || "").trim().length > 0
    : true;

  const canContinue = hasEnrichment
    ? allNonEnrichmentFilled && brandNameFilled && competitorTags.length > 0
    : allNonEnrichmentFilled && brandNameFilled;

  const handleConfirm = () => {
    onConfirm(parameterValues, { competitors: competitorTags });
  };

  return (
    <Modal
      title={
        <div>
          <h3 className="text-lg font-semibold mb-1">{template?.title}</h3>
          <p className="text-sm text-gray-500 font-normal">
            Configure competitor blocking for your brand
          </p>
        </div>
      }
      open={visible}
      onCancel={onCancel}
      width={700}
      footer={[
        <Button key="cancel" variant="secondary" onClick={onCancel} disabled={isLoading}>
          Cancel
        </Button>,
        <Button
          key="confirm"
          onClick={handleConfirm}
          loading={isLoading}
          disabled={!canContinue || isLoading}
        >
          {isLoading ? "Creating guardrails..." : "Continue"}
        </Button>,
      ]}
    >
      <div className="py-4 space-y-4">
        {nonEnrichmentParams.map((param) => (
          <div key={param.name}>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              {param.label}
              {param.required && <span className="text-red-500 ml-1">*</span>}
            </label>
            <TextInput
              placeholder={param.placeholder || ""}
              value={parameterValues[param.name] || ""}
              onChange={(e) =>
                setParameterValues((prev) => ({
                  ...prev,
                  [param.name]: e.target.value,
                }))
              }
            />
          </div>
        ))}

        {hasEnrichment && (
          <>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Competitor Discovery
              </label>
              <Radio.Group
                value={competitorMode}
                onChange={(e) => setCompetitorMode(e.target.value)}
                className="w-full"
              >
                <div className="flex gap-3">
                  <Radio.Button value="ai" className="flex-1 text-center">
                    ✨ Use AI
                  </Radio.Button>
                  <Radio.Button value="manual" className="flex-1 text-center">
                    Enter Manually
                  </Radio.Button>
                </div>
              </Radio.Group>
            </div>

            {/* Brand Name */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Your Brand Name
                <span className="text-red-500 ml-1">*</span>
              </label>
              <TextInput
                placeholder="e.g. Acme Airlines"
                value={parameterValues[enrichmentParam || "brand_name"] || ""}
                onChange={(e) =>
                  setParameterValues((prev) => ({
                    ...prev,
                    [enrichmentParam || "brand_name"]: e.target.value,
                  }))
                }
              />
            </div>

            {competitorMode === "ai" && (
              <>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Select Model
                    <span className="text-red-500 ml-1">*</span>
                  </label>
                  <Select
                    placeholder="Select a model to generate names"
                    value={selectedModel}
                    onChange={(value) => setSelectedModel(value)}
                    loading={isLoadingModels}
                    showSearch
                    className="w-full"
                    options={availableModels.map((m) => ({ label: m, value: m }))}
                    filterOption={(input, option) =>
                      (option?.label ?? "").toLowerCase().includes(input.toLowerCase())
                    }
                  />
                </div>

                <Button
                  onClick={handleGenerateNames}
                  loading={isGenerating}
                  disabled={!selectedModel || !brandNameFilled || isGenerating}
                  className="w-full"
                >
                  {isGenerating ? "✨ Generating names..." : "✨ Generate Competitor Names"}
                </Button>
              </>
            )}

            {/* Competitor Tags */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Competitor Names
                {competitorTags.length > 0 && (
                  <span className="text-gray-400 font-normal ml-2">
                    ({competitorTags.length})
                  </span>
                )}
              </label>
              <Select
                mode="tags"
                style={{ width: "100%" }}
                placeholder="Type a name and press Enter to add"
                value={competitorTags}
                onChange={(values) => setCompetitorTags(values)}
                tokenSeparators={[","]}
                open={false}
                suffixIcon={null}
              />
              <p className="text-xs text-gray-500 mt-1">
                Type a name and press Enter to add. Click ✕ to remove.
              </p>
              {statusMessage && (
                <div className="flex items-center gap-2 mt-2 p-2 bg-blue-50 rounded border border-blue-100">
                  <Spin size="small" />
                  <span className="text-xs text-blue-700">{statusMessage}</span>
                </div>
              )}
              {Object.keys(variationsMap).length > 0 && !statusMessage && (
                <p className="text-xs text-green-600 mt-1">
                  ✓ {Object.values(variationsMap).flat().length} alternate spellings & variations auto-generated for guardrail matching
                </p>
              )}
            </div>

            {/* Refinement input — shown after initial generation in AI mode */}
            {competitorMode === "ai" && hasGenerated && competitorTags.length > 0 && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Refine List
                </label>
                <div className="flex gap-2">
                  <TextInput
                    placeholder="e.g. add 10 more from Asia, increase to 50 total..."
                    value={refinementInput}
                    onChange={(e) => setRefinementInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && refinementInput.trim() && !isRefining) {
                        handleRefine();
                      }
                    }}
                    disabled={isRefining}
                  />
                  <Button
                    onClick={handleRefine}
                    loading={isRefining}
                    disabled={!refinementInput.trim() || isRefining}
                    size="xs"
                  >
                    {isRefining ? "..." : "Send"}
                  </Button>
                </div>
                <p className="text-xs text-gray-400 mt-1">
                  Give instructions to add, remove, or change competitors. Press Enter to send.
                </p>
              </div>
            )}
          </>
        )}

        {!hasEnrichment &&
          parameters.map((param) => (
            <div key={param.name}>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                {param.label}
                {param.required && <span className="text-red-500 ml-1">*</span>}
              </label>
              <TextInput
                placeholder={param.placeholder || ""}
                value={parameterValues[param.name] || ""}
                onChange={(e) =>
                  setParameterValues((prev) => ({
                    ...prev,
                    [param.name]: e.target.value,
                  }))
                }
              />
            </div>
          ))}
      </div>
    </Modal>
  );
};

export default TemplateParameterModal;
