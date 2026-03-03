import { useState } from "react";
import { Typography, Descriptions, Card, Tag, Tabs, Alert, Collapse, Radio, Space, Spin } from "antd";
import moment from "moment";
import { LogEntry } from "../columns";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import GuardrailViewer from "../GuardrailViewer/GuardrailViewer";
import { CostBreakdownViewer } from "../CostBreakdownViewer";
import { ConfigInfoMessage } from "../ConfigInfoMessage";
import { VectorStoreViewer } from "../VectorStoreViewer";
import { TruncatedValue } from "./TruncatedValue";
import { TokenFlow } from "./TokenFlow";
import { JsonViewer } from "./JsonViewer";
import {
  formatData,
  checkHasMessages,
  checkHasResponse,
  normalizeGuardrailEntries,
  calculateTotalMaskedEntities,
  getGuardrailLabel,
  checkHasVectorStoreData,
} from "./utils";
import {
  DRAWER_CONTENT_PADDING,
  API_BASE_MAX_WIDTH,
  METADATA_MAX_HEIGHT,
  TAB_REQUEST,
  TAB_RESPONSE,
  FONT_SIZE_SMALL,
  FONT_FAMILY_MONO,
  SPACING_XLARGE,
  SPACING_MEDIUM,
} from "./constants";
import { ToolsSection } from "../ToolsSection";
import { PrettyMessagesView } from "./PrettyMessagesView";

const { Text } = Typography;

export interface LogDetailContentProps {
  logEntry: LogEntry;
  onOpenSettings?: () => void;
  /** When true, log details (messages/response) are still being lazy-loaded. */
  isLoadingDetails?: boolean;
  accessToken?: string | null;
}

/**
 * The scrollable detail content for a single log entry.
 * Renders request details, metrics, cost breakdown, request/response,
 * guardrails, vector store data, and metadata.
 *
 * Designed to be placed inside LogDetailsDrawer's right panel so it can
 * be reused for both single-log and session-mode views.
 */
export function LogDetailContent({ logEntry, onOpenSettings, isLoadingDetails = false, accessToken }: LogDetailContentProps) {
  const metadata = logEntry.metadata || {};
  const hasError = metadata.status === "failure";
  const errorInfo = hasError ? metadata.error_information : null;

  const hasMessages = checkHasMessages(logEntry.messages);
  const hasResponse = checkHasResponse(logEntry.response);
  // Don't show "missing data" warning while details are still loading
  const missingData = !hasMessages && !hasResponse && !hasError && !isLoadingDetails;

  // Guardrail data
  const guardrailInfo = metadata?.guardrail_information;
  const guardrailEntries = normalizeGuardrailEntries(guardrailInfo);
  const hasGuardrailData = guardrailEntries.length > 0;
  const totalMaskedEntities = calculateTotalMaskedEntities(guardrailEntries);
  const primaryGuardrailLabel = getGuardrailLabel(guardrailEntries);

  // Vector store data
  const hasVectorStoreData = checkHasVectorStoreData(metadata);

  const getRawRequest = () => {
    return formatData(logEntry.proxy_server_request || logEntry.messages);
  };

  const getFormattedResponse = () => {
    if (hasError && errorInfo) {
      return {
        error: {
          message: errorInfo.error_message || "An error occurred",
          type: errorInfo.error_class || "error",
          code: errorInfo.error_code || "unknown",
          param: null,
        },
      };
    }
    return formatData(logEntry.response);
  };

  return (
    <div style={{ padding: `${DRAWER_CONTENT_PADDING} ${DRAWER_CONTENT_PADDING} 0` }}>
      {/* Error Alert */}
      {hasError && errorInfo && (
        <Alert
          type="error"
          showIcon
          message="Request Failed"
          description={<ErrorDescription errorInfo={errorInfo} />}
          className="mb-6"
        />
      )}

      {/* Tags */}
      {logEntry.request_tags && Object.keys(logEntry.request_tags).length > 0 && (
        <TagsSection tags={logEntry.request_tags} />
      )}

      {/* Request Details */}
      <div className="bg-white rounded-lg shadow w-full max-w-full overflow-hidden mb-6">
        <Card title="Request Details" size="small" bordered={false} style={{ marginBottom: 0 }}>
          <Descriptions column={2} size="small">
            <Descriptions.Item label="Model">{logEntry.model}</Descriptions.Item>
            <Descriptions.Item label="Provider">{logEntry.custom_llm_provider || "-"}</Descriptions.Item>
            <Descriptions.Item label="Call Type">{logEntry.call_type}</Descriptions.Item>
            <Descriptions.Item label="Model ID">
              <TruncatedValue value={logEntry.model_id} />
            </Descriptions.Item>
            <Descriptions.Item label="API Base">
              <TruncatedValue value={logEntry.api_base} maxWidth={API_BASE_MAX_WIDTH} />
            </Descriptions.Item>
            {logEntry.requester_ip_address && (
              <Descriptions.Item label="IP Address">{logEntry.requester_ip_address}</Descriptions.Item>
            )}
            {hasGuardrailData && (
              <Descriptions.Item label="Guardrail">
                <GuardrailLabel label={primaryGuardrailLabel} maskedCount={totalMaskedEntities} />
              </Descriptions.Item>
            )}
          </Descriptions>
        </Card>
      </div>

      {/* Routing Info */}
      <RoutingSection metadata={metadata} modelGroup={logEntry.model_group} selectedModel={logEntry.model} selectedModelId={logEntry.model_id} />

      {/* Metrics */}
      <MetricsSection logEntry={logEntry} metadata={metadata} />

      {/* Cost Breakdown */}
      <CostBreakdownViewer
        costBreakdown={metadata?.cost_breakdown}
        totalSpend={logEntry.spend ?? 0}
        promptTokens={logEntry.prompt_tokens}
        completionTokens={logEntry.completion_tokens}
        cacheHit={logEntry.cache_hit}
      />

      {/* Tools */}
      <ToolsSection log={logEntry} />

      {/* Configuration Info Message */}
      {missingData && (
        <div className="mb-6">
          <ConfigInfoMessage show={missingData} onOpenSettings={onOpenSettings} />
        </div>
      )}

      {/* Request/Response JSON */}
      {isLoadingDetails ? (
        <div className="bg-white rounded-lg shadow w-full max-w-full overflow-hidden mb-6 p-8 text-center">
          <Spin size="default" />
          <div style={{ marginTop: 8, color: "#999" }}>Loading request &amp; response data...</div>
        </div>
      ) : (
        <RequestResponseSection
          hasResponse={hasResponse}
          hasError={hasError}
          getRawRequest={getRawRequest}
          getFormattedResponse={getFormattedResponse}
          logEntry={logEntry}
        />
      )}

      {/* Guardrail Data */}
      {hasGuardrailData && (
        <div id="guardrail-section">
          <GuardrailViewer
            data={guardrailInfo}
            accessToken={accessToken ?? null}
            logEntry={{
              request_id: logEntry.request_id,
              user: logEntry.user,
              model: logEntry.model,
              startTime: logEntry.startTime,
              metadata: logEntry.metadata,
            }}
          />
        </div>
      )}

      {/* Vector Store Data */}
      {hasVectorStoreData && <VectorStoreViewer data={metadata.vector_store_request_metadata} />}

      {/* Metadata */}
      {logEntry.metadata && Object.keys(logEntry.metadata).length > 0 && (
        <MetadataSection metadata={logEntry.metadata} />
      )}

      {/* Bottom spacing */}
      <div style={{ height: DRAWER_CONTENT_PADDING }} />
    </div>
  );
}

// ============================================================================
// Helper Components
// ============================================================================

function ErrorDescription({ errorInfo }: { errorInfo: any }) {
  return (
    <div>
      {errorInfo.error_code && (
        <div>
          <Text strong>Error Code:</Text> {errorInfo.error_code}
        </div>
      )}
      {errorInfo.error_message && (
        <div>
          <Text strong>Message:</Text> {errorInfo.error_message}
        </div>
      )}
    </div>
  );
}

function TagsSection({ tags }: { tags: Record<string, any> }) {
  return (
    <div className="bg-white rounded-lg shadow w-full max-w-full overflow-hidden p-4 mb-6">
      <Text strong style={{ display: "block", marginBottom: 8, fontSize: 16 }}>
        Tags
      </Text>
      <Space size={SPACING_MEDIUM} wrap>
        {Object.entries(tags).map(([key, value]) => (
          <Tag key={key}>
            {key}: {String(value)}
          </Tag>
        ))}
      </Space>
    </div>
  );
}

function GuardrailLabel({ label, maskedCount }: { label: string; maskedCount: number }) {
  const handleClick = () => {
    const el = document.getElementById("guardrail-section");
    if (el) el.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <Space size={SPACING_MEDIUM}>
      <a onClick={handleClick} style={{ cursor: "pointer" }}>{label}</a>
      {maskedCount > 0 && (
        <Tag color="blue">
          {maskedCount} masked
        </Tag>
      )}
    </Space>
  );
}

function MetricsSection({ logEntry, metadata }: { logEntry: LogEntry; metadata: Record<string, any> }) {
  const hasCacheActivity =
    logEntry.cache_hit ||
    (metadata?.additional_usage_values?.cache_read_input_tokens &&
      metadata.additional_usage_values.cache_read_input_tokens > 0);

  const cacheHitValue = String(logEntry.cache_hit ?? "None");
  const cacheHitColor =
    cacheHitValue.toLowerCase() === "true"
      ? "green"
      : cacheHitValue.toLowerCase() === "false"
        ? "red"
        : "default";

  return (
    <div className="bg-white rounded-lg shadow w-full max-w-full overflow-hidden mb-6">
      <Card title="Metrics" size="small" style={{ marginBottom: 0 }}>
        <Descriptions column={2} size="small">
          <Descriptions.Item label="Tokens">
            <TokenFlow
              prompt={logEntry.prompt_tokens}
              completion={logEntry.completion_tokens}
              total={logEntry.total_tokens}
            />
          </Descriptions.Item>
          <Descriptions.Item label="Cost">${formatNumberWithCommas(logEntry.spend || 0, 8)}</Descriptions.Item>
          <Descriptions.Item label="Duration">{logEntry.request_duration_ms != null ? (logEntry.request_duration_ms / 1000).toFixed(3) : "-"} s</Descriptions.Item>

          {hasCacheActivity && (
            <>
              <Descriptions.Item label="Cache Hit">
                <Tag color={cacheHitColor}>{cacheHitValue}</Tag>
              </Descriptions.Item>
              {metadata?.additional_usage_values?.cache_read_input_tokens > 0 && (
                <Descriptions.Item label="Cache Read Tokens">
                  {formatNumberWithCommas(metadata.additional_usage_values.cache_read_input_tokens)}
                </Descriptions.Item>
              )}
              {metadata?.additional_usage_values?.cache_creation_input_tokens > 0 && (
                <Descriptions.Item label="Cache Creation Tokens">
                  {formatNumberWithCommas(metadata.additional_usage_values.cache_creation_input_tokens)}
                </Descriptions.Item>
              )}
            </>
          )}

          {metadata?.litellm_overhead_time_ms !== undefined && metadata.litellm_overhead_time_ms !== null && (
            <Descriptions.Item label="LiteLLM Overhead">
              {metadata.litellm_overhead_time_ms.toFixed(2)} ms
            </Descriptions.Item>
          )}

          <Descriptions.Item label="Retries">
            {metadata?.attempted_retries !== undefined && metadata?.attempted_retries !== null
              ? metadata.attempted_retries > 0
                ? <>{metadata.attempted_retries}{metadata.max_retries !== undefined && metadata.max_retries !== null ? ` / ${metadata.max_retries}` : ''}</>
                : <Tag color="green">None</Tag>
              : "-"}
          </Descriptions.Item>

          <Descriptions.Item label="Start Time">
            {moment(logEntry.startTime).format("YYYY-MM-DDTHH:mm:ss.SSS[Z]")}
          </Descriptions.Item>
          <Descriptions.Item label="End Time">
            {moment(logEntry.endTime).format("YYYY-MM-DDTHH:mm:ss.SSS[Z]")}
          </Descriptions.Item>
        </Descriptions>
      </Card>
    </div>
  );
}

interface RequestResponseSectionProps {
  hasResponse: boolean;
  hasError: boolean;
  getRawRequest: () => any;
  getFormattedResponse: () => any;
  logEntry: LogEntry;
}

function RequestResponseSection({
  hasResponse,
  hasError,
  getRawRequest,
  getFormattedResponse,
  logEntry,
}: RequestResponseSectionProps) {
  const [activeTab, setActiveTab] = useState<typeof TAB_REQUEST | typeof TAB_RESPONSE>(TAB_REQUEST);
  const [viewMode, setViewMode] = useState<'pretty' | 'json'>('pretty');

  const getCopyText = () => {
    const data = activeTab === TAB_REQUEST ? getRawRequest() : getFormattedResponse();
    return JSON.stringify(data, null, 2);
  };

  const totalSpend = logEntry.spend ?? 0;
  const promptTokens = logEntry.prompt_tokens || 0;
  const completionTokens = logEntry.completion_tokens || 0;
  const totalTokens = promptTokens + completionTokens;
  const costBreakdown = logEntry.metadata?.cost_breakdown;
  const useCostBreakdown =
    costBreakdown?.input_cost !== undefined &&
    costBreakdown?.output_cost !== undefined;
  const inputCost = useCostBreakdown
    ? (costBreakdown!.input_cost ?? 0)
    : totalTokens > 0
      ? (totalSpend * promptTokens) / totalTokens
      : 0;
  const outputCost = useCostBreakdown
    ? (costBreakdown!.output_cost ?? 0)
    : totalTokens > 0
      ? (totalSpend * completionTokens) / totalTokens
      : 0;

  return (
    <div className="bg-white rounded-lg shadow w-full max-w-full overflow-hidden mb-6">
      <Collapse
        defaultActiveKey={["1"]}
        expandIconPosition="start"
        items={[
          {
            key: "1",
            label: (
              <div
                style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}
                onClick={(e) => {
                  const target = e.target as HTMLElement;
                  if (target.closest('.ant-radio-group')) {
                    e.stopPropagation();
                  }
                }}
              >
                <h3 className="text-lg font-medium text-gray-900" style={{ margin: 0 }}>Request & Response</h3>
                <Radio.Group
                  size="small"
                  value={viewMode}
                  onChange={(e) => setViewMode(e.target.value)}
                >
                  <Radio.Button value="pretty">Pretty</Radio.Button>
                  <Radio.Button value="json">JSON</Radio.Button>
                </Radio.Group>
              </div>
            ),
            children: (
              <div>
                {viewMode === 'pretty' ? (
                  <PrettyMessagesView
                    request={getRawRequest()}
                    response={getFormattedResponse()}
                    metrics={{
                      prompt_tokens: promptTokens,
                      completion_tokens: completionTokens,
                      input_cost: inputCost,
                      output_cost: outputCost,
                    }}
                  />
                ) : (
                  <Tabs
                    activeKey={activeTab}
                    onChange={(key) => setActiveTab(key as typeof TAB_REQUEST | typeof TAB_RESPONSE)}
                    tabBarExtraContent={
                      <Text
                        copyable={{
                          text: getCopyText(),
                          tooltips: ["Copy JSON", "Copied!"]
                        }}
                        disabled={activeTab === TAB_RESPONSE && !hasResponse && !hasError}
                      />
                    }
                    items={[
                      {
                        key: TAB_REQUEST,
                        label: "Request",
                        children: (
                          <div style={{ paddingTop: SPACING_XLARGE, paddingBottom: SPACING_XLARGE }}>
                            <JsonViewer data={getRawRequest()} mode="formatted" />
                          </div>
                        ),
                      },
                      {
                        key: TAB_RESPONSE,
                        label: "Response",
                        children: (
                          <div style={{ paddingTop: SPACING_XLARGE, paddingBottom: SPACING_XLARGE }}>
                            {hasResponse || hasError ? (
                              <JsonViewer data={getFormattedResponse()} mode="formatted" />
                            ) : (
                              <div style={{ textAlign: "center", padding: 20, color: "#999", fontStyle: "italic" }}>
                                Response data not available
                              </div>
                            )}
                          </div>
                        ),
                      },
                    ]}
                  />
                )}
              </div>
            ),
          },
        ]}
      />
    </div>
  );
}

export function GuardrailJumpLink({ guardrailEntries }: { guardrailEntries: any[] }) {
  const allPassed = guardrailEntries.every((e) => {
    const status = e?.guardrail_status || e?.status;
    return status === "pass" || status === "passed" || status === "success";
  });

  const handleClick = () => {
    const el = document.getElementById("guardrail-section");
    if (el) el.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <div style={{ textAlign: "left", marginBottom: 12 }}>
      <div
        onClick={handleClick}
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: 6,
          padding: "4px 12px",
          borderRadius: 16,
          cursor: "pointer",
          fontSize: 13,
          fontWeight: 500,
          backgroundColor: allPassed ? "#f0fdf4" : "#fef2f2",
          color: allPassed ? "#15803d" : "#b91c1c",
          border: `1px solid ${allPassed ? "#bbf7d0" : "#fecaca"}`,
        }}
      >
        {allPassed ? "\u2713" : "\u2717"} {guardrailEntries.length} guardrail{guardrailEntries.length !== 1 ? "s" : ""} evaluated
        <span style={{ fontSize: 11, opacity: 0.7 }}>{"\u2193"}</span>
      </div>
    </div>
  );
}

const STRATEGY_LABELS: Record<string, string> = {
  "simple-shuffle": "Random",
  "latency-based-routing": "Lowest Latency",
  "usage-based-routing-v2": "Usage-Based",
  "cost-based-routing": "Lowest Cost",
  "least-busy": "Least Busy",
  "priority-failover": "Ordered Fallback",
  "weighted-round-robin": "Weighted Round Robin",
  "complexity-router": "Complexity Router",
};

function RoutingSection({
  metadata,
  modelGroup,
  selectedModel,
  selectedModelId,
}: {
  metadata: Record<string, any>;
  modelGroup?: string;
  selectedModel: string;
  selectedModelId: string;
}) {
  const modelGroupSize = metadata?.model_group_size;
  const routingStrategy = metadata?.routing_strategy;
  const candidates: Array<{ model_id?: string; model_name?: string; litellm_model?: string; priority?: number }> =
    metadata?.model_group_candidates || [];

  if (!modelGroup) return null;

  const isRoutingGroup = modelGroup !== selectedModel;
  const strategyLabel = (routingStrategy && STRATEGY_LABELS[routingStrategy]) || routingStrategy;
  const isOrdered = routingStrategy === "priority-failover";
  const sortedCandidates = isOrdered
    ? [...candidates].sort((a, b) => (a.priority ?? 999) - (b.priority ?? 999))
    : candidates;

  return (
    <div className="bg-white rounded-lg shadow w-full max-w-full overflow-hidden mb-6">
      <Card
        title={
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span>Routing</span>
            {isRoutingGroup && (
              <Tag color="blue" style={{ margin: 0, fontSize: 11 }}>Routing Group</Tag>
            )}
          </div>
        }
        size="small"
        bordered={false}
        style={{ marginBottom: 0 }}
      >
        <Descriptions column={2} size="small">
          <Descriptions.Item label="Model Group">
            <Text strong>{modelGroup}</Text>
          </Descriptions.Item>
          <Descriptions.Item label="Selected">
            <Text>{selectedModel}</Text>
            {selectedModelId && (
              <Text type="secondary" style={{ fontSize: 11, marginLeft: 4 }}>
                ({selectedModelId.slice(0, 8)}…)
              </Text>
            )}
          </Descriptions.Item>
          {routingStrategy && (
            <Descriptions.Item label="Strategy">
              <Tag style={{ margin: 0 }}>{strategyLabel}</Tag>
            </Descriptions.Item>
          )}
          {modelGroupSize != null && (
            <Descriptions.Item label="Candidates">
              {modelGroupSize} deployment{modelGroupSize !== 1 ? "s" : ""}
            </Descriptions.Item>
          )}
        </Descriptions>

        {/* Candidate list */}
        {sortedCandidates.length > 0 && (
          <div style={{ marginTop: 12 }}>
            <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 8, fontWeight: 500 }}>
              {isOrdered ? "Fallback Order" : "Candidates Considered"}
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
              {sortedCandidates.map((c, idx) => {
                const isSelected = c.model_id === selectedModelId;
                const label = c.litellm_model || c.model_name || c.model_id || "unknown";
                return (
                  <div key={c.model_id || idx}>
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 10,
                        padding: "8px 12px",
                        borderRadius: 8,
                        backgroundColor: isSelected ? "#eff6ff" : "#fafafa",
                        border: isSelected ? "1px solid #bfdbfe" : "1px solid #f3f4f6",
                      }}
                    >
                      {isOrdered && (
                        <div style={{
                          width: 22, height: 22, borderRadius: "50%",
                          backgroundColor: isSelected ? "#3b82f6" : "#e5e7eb",
                          color: isSelected ? "#fff" : "#6b7280",
                          display: "flex", alignItems: "center", justifyContent: "center",
                          fontSize: 11, fontWeight: 600, flexShrink: 0,
                        }}>
                          {idx + 1}
                        </div>
                      )}
                      {!isOrdered && (
                        <div style={{
                          width: 6, height: 6, borderRadius: "50%",
                          backgroundColor: isSelected ? "#3b82f6" : "#d1d5db",
                          flexShrink: 0,
                        }} />
                      )}
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <Text style={{ fontSize: 13, fontWeight: isSelected ? 600 : 400 }}>
                          {label}
                        </Text>
                        {c.model_id && (
                          <Text type="secondary" style={{ fontSize: 10, marginLeft: 6 }}>
                            {c.model_id.slice(0, 8)}…
                          </Text>
                        )}
                      </div>
                      {isSelected && (
                        <Tag color="green" style={{ margin: 0, fontSize: 10 }}>selected</Tag>
                      )}
                      {isOrdered && !isSelected && idx === 0 && (
                        <Tag style={{ margin: 0, fontSize: 10, color: "#9ca3af", borderColor: "#e5e7eb" }}>primary</Tag>
                      )}
                    </div>
                    {/* Connector line between items */}
                    {isOrdered && idx < sortedCandidates.length - 1 && (
                      <div style={{ display: "flex", alignItems: "center", paddingLeft: 22, height: 16 }}>
                        <div style={{ width: 1, height: "100%", backgroundColor: "#e5e7eb", marginLeft: 0 }} />
                        <span style={{ fontSize: 9, color: "#d1d5db", marginLeft: 8 }}>↓ fallback</span>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Explanation */}
        {modelGroupSize != null && modelGroupSize > 1 && (
          <div style={{ marginTop: 12, padding: "8px 12px", backgroundColor: "#f9fafb", borderRadius: 8, border: "1px solid #f3f4f6" }}>
            <div style={{ fontSize: 12, color: "#374151", lineHeight: 1.5 }}>
              {routingStrategy === "simple-shuffle" && (
                <>Randomly selected from {modelGroupSize} available deployments.</>
              )}
              {routingStrategy === "latency-based-routing" && (
                <>Picked the lowest-latency deployment from {modelGroupSize} candidates.</>
              )}
              {routingStrategy === "usage-based-routing-v2" && (
                <>Picked the deployment with the most available TPM/RPM capacity.</>
              )}
              {routingStrategy === "cost-based-routing" && (
                <>Picked the lowest-cost deployment from {modelGroupSize} candidates.</>
              )}
              {routingStrategy === "least-busy" && (
                <>Picked the deployment with the fewest active requests.</>
              )}
              {routingStrategy === "priority-failover" && (
                <>Tried deployments in priority order. {
                  sortedCandidates.length > 0 && sortedCandidates[0]?.model_id === selectedModelId
                    ? "Primary deployment was healthy."
                    : `Primary was unavailable — fell back to priority ${(sortedCandidates.findIndex(c => c.model_id === selectedModelId) + 1) || "?"}.`
                }</>
              )}
              {!routingStrategy && (
                <>Router selected <strong>{selectedModel}</strong> from {modelGroupSize} deployments.</>
              )}
              {routingStrategy && !["simple-shuffle", "latency-based-routing", "usage-based-routing-v2", "cost-based-routing", "least-busy", "priority-failover"].includes(routingStrategy) && (
                <>Strategy: {strategyLabel}. Selected from {modelGroupSize} candidates.</>
              )}
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}

function MetadataSection({ metadata }: { metadata: Record<string, any> }) {
  return (
    <div className="bg-white rounded-lg shadow w-full max-w-full overflow-hidden mb-6">
      <Collapse
        defaultActiveKey={["1"]}
        expandIconPosition="start"
        items={[
          {
            key: "1",
            label: <h3 className="text-lg font-medium text-gray-900">Metadata</h3>,
            children: (
              <div>
                <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 8 }}>
                  <Text
                    copyable={{
                      text: JSON.stringify(metadata, null, 2),
                      tooltips: ["Copy Metadata", "Copied!"]
                    }}
                  />
                </div>
                <pre
                  style={{
                    maxHeight: METADATA_MAX_HEIGHT,
                    overflowY: "auto",
                    fontSize: FONT_SIZE_SMALL,
                    fontFamily: FONT_FAMILY_MONO,
                    whiteSpace: "pre-wrap",
                    wordBreak: "break-all",
                    margin: 0,
                  }}
                >
                  {JSON.stringify(metadata, null, 2)}
                </pre>
              </div>
            ),
          },
        ]}
      />
    </div>
  );
}
