import React, { useState } from "react";
import { Switch, TextInput, Select, SelectItem, Button, Badge } from "@tremor/react";

export interface ContentRoutingConfig {
  enabled: boolean;
  classifier: "rule_based" | "embedding_similarity" | "external_model";
  default_model?: string;
  confidence_threshold?: number;
  embedding_model?: string;
  external_classifier_url?: string;
}

interface ContentRouteTestResult {
  matched_preference: string;
  matched_model: string;
  confidence: number;
  classifier: string;
  all_scores?: Record<string, number>;
}

interface ContentRoutingConfigurationProps {
  config: ContentRoutingConfig;
  onChange: (config: ContentRoutingConfig) => void;
  accessToken: string | null;
}

const DEFAULT_CONFIG: ContentRoutingConfig = {
  enabled: false,
  classifier: "rule_based",
  default_model: "",
  confidence_threshold: 0.1,
};

const CLASSIFIER_DESCRIPTIONS: Record<string, string> = {
  rule_based:
    "TF-IDF keyword matching — zero latency, no extra API calls. Best starting point.",
  embedding_similarity:
    "Semantic embedding similarity using litellm.aembedding(). More flexible but adds one embedding call per request.",
  external_model:
    "Delegates to an external HTTP classifier (e.g. Arch-Router). Fully pluggable.",
};

const ContentRoutingConfiguration: React.FC<ContentRoutingConfigurationProps> = ({
  config,
  onChange,
  accessToken,
}) => {
  const [testPrompt, setTestPrompt] = useState("");
  const [testResult, setTestResult] = useState<ContentRouteTestResult | null>(null);
  const [testLoading, setTestLoading] = useState(false);
  const [testError, setTestError] = useState<string | null>(null);
  const [showTestPanel, setShowTestPanel] = useState(false);

  const effective = { ...DEFAULT_CONFIG, ...config };

  const update = (patch: Partial<ContentRoutingConfig>) =>
    onChange({ ...effective, ...patch });

  const runTest = async () => {
    if (!testPrompt.trim() || !accessToken) return;
    setTestLoading(true);
    setTestError(null);
    setTestResult(null);
    try {
      const resp = await fetch("/utils/content_route_test", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${accessToken}`,
        },
        body: JSON.stringify({ prompt: testPrompt }),
      });
      if (!resp.ok) {
        const data = await resp.json().catch(() => ({}));
        throw new Error(data?.detail ?? `HTTP ${resp.status}`);
      }
      setTestResult(await resp.json());
    } catch (e: any) {
      setTestError(e?.message ?? "Unknown error");
    } finally {
      setTestLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="max-w-3xl">
        <h3 className="text-sm font-medium text-gray-900">Content-Aware Routing</h3>
        <p className="text-xs text-gray-500 mt-1">
          Classify prompt content and route to the best-matched model based on{" "}
          <code className="bg-gray-100 px-1 rounded">routing_preferences</code> configured
          per model. Runs before infrastructure routing (latency, cost, etc.).
        </p>
      </div>

      {/* Enable toggle */}
      <div className="flex items-center gap-3">
        <Switch
          id="content-routing-enabled"
          checked={effective.enabled}
          onChange={(v) => update({ enabled: v })}
        />
        <label htmlFor="content-routing-enabled" className="text-sm text-gray-700 cursor-pointer">
          Enable content-aware routing
        </label>
      </div>

      {effective.enabled && (
        <div className="space-y-6 pl-0">
          {/* Classifier */}
          <div className="space-y-2 max-w-md">
            <label className="block text-xs font-medium text-gray-700 uppercase tracking-wide">
              Classifier
            </label>
            <Select
              value={effective.classifier}
              onValueChange={(v) =>
                update({ classifier: v as ContentRoutingConfig["classifier"] })
              }
            >
              <SelectItem value="rule_based">Rule-Based (TF-IDF)</SelectItem>
              <SelectItem value="embedding_similarity">Embedding Similarity</SelectItem>
              <SelectItem value="external_model">External Model</SelectItem>
            </Select>
            <p className="text-xs text-gray-500">
              {CLASSIFIER_DESCRIPTIONS[effective.classifier]}
            </p>
          </div>

          {/* Common fields */}
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <div className="space-y-2">
              <label className="block text-xs font-medium text-gray-700 uppercase tracking-wide">
                Default Model
              </label>
              <p className="text-xs text-gray-500">
                Fallback when no preference matches above the confidence threshold.
              </p>
              <TextInput
                placeholder="e.g. gpt-4o"
                value={effective.default_model ?? ""}
                onChange={(e) => update({ default_model: e.target.value })}
              />
            </div>

            <div className="space-y-2">
              <label className="block text-xs font-medium text-gray-700 uppercase tracking-wide">
                Confidence Threshold
              </label>
              <p className="text-xs text-gray-500">
                Minimum score (0.0–1.0) required to route to a matched model.
              </p>
              <TextInput
                placeholder="0.1"
                value={(effective.confidence_threshold ?? 0.1).toString()}
                onChange={(e) => {
                  const n = parseFloat(e.target.value);
                  if (!isNaN(n)) update({ confidence_threshold: n });
                }}
              />
            </div>
          </div>

          {/* Embedding-specific */}
          {effective.classifier === "embedding_similarity" && (
            <div className="space-y-2 max-w-md">
              <label className="block text-xs font-medium text-gray-700 uppercase tracking-wide">
                Embedding Model
              </label>
              <p className="text-xs text-gray-500">
                Model used to embed preference descriptions and incoming prompts.
              </p>
              <TextInput
                placeholder="text-embedding-3-small"
                value={effective.embedding_model ?? ""}
                onChange={(e) => update({ embedding_model: e.target.value })}
              />
            </div>
          )}

          {/* External model-specific */}
          {effective.classifier === "external_model" && (
            <div className="space-y-2 max-w-lg">
              <label className="block text-xs font-medium text-gray-700 uppercase tracking-wide">
                External Classifier URL
              </label>
              <p className="text-xs text-gray-500">
                POST endpoint that accepts{" "}
                <code className="bg-gray-100 px-1 rounded">
                  {"{ prompt, preferences }"}
                </code>{" "}
                and returns{" "}
                <code className="bg-gray-100 px-1 rounded">
                  {"{ matched_preference, model, confidence }"}
                </code>
                . Compatible with Arch-Router.
              </p>
              <TextInput
                placeholder="http://arch-router-host/classify"
                value={effective.external_classifier_url ?? ""}
                onChange={(e) => update({ external_classifier_url: e.target.value })}
              />
            </div>
          )}

          {/* Test routing panel */}
          <div className="border border-gray-200 rounded-lg p-4 space-y-3 max-w-2xl">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-900">Test Content Routing</p>
                <p className="text-xs text-gray-500">
                  Classify a prompt without making an LLM call.
                </p>
              </div>
              <Button
                size="xs"
                variant="secondary"
                onClick={() => setShowTestPanel((v) => !v)}
              >
                {showTestPanel ? "Hide" : "Open tester"}
              </Button>
            </div>

            {showTestPanel && (
              <div className="space-y-3">
                <TextInput
                  placeholder="e.g. write a Python function to sort a list"
                  value={testPrompt}
                  onChange={(e) => setTestPrompt(e.target.value)}
                />
                <Button
                  size="xs"
                  loading={testLoading}
                  disabled={!testPrompt.trim()}
                  onClick={runTest}
                >
                  Run classification
                </Button>

                {testError && (
                  <p className="text-xs text-red-600">{testError}</p>
                )}

                {testResult && (
                  <div className="bg-gray-50 rounded p-3 space-y-2 text-xs">
                    <div className="flex items-center gap-2">
                      <span className="text-gray-500">Matched preference:</span>
                      <Badge color="blue" size="xs">{testResult.matched_preference}</Badge>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-gray-500">Routed to:</span>
                      <span className="font-mono font-medium">{testResult.matched_model}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-gray-500">Confidence:</span>
                      <span className="font-mono">{testResult.confidence.toFixed(4)}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-gray-500">Classifier:</span>
                      <span className="font-mono">{testResult.classifier}</span>
                    </div>
                    {testResult.all_scores && Object.keys(testResult.all_scores).length > 0 && (
                      <details className="mt-1">
                        <summary className="cursor-pointer text-gray-500 hover:text-gray-700">
                          All scores
                        </summary>
                        <div className="mt-1 space-y-1 pl-2">
                          {Object.entries(testResult.all_scores)
                            .sort(([, a], [, b]) => b - a)
                            .map(([key, score]) => (
                              <div key={key} className="flex justify-between font-mono">
                                <span className="text-gray-600">{key}</span>
                                <span>{score.toFixed(4)}</span>
                              </div>
                            ))}
                        </div>
                      </details>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ContentRoutingConfiguration;
