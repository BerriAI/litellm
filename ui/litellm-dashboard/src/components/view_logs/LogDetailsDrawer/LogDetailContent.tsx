import { useState } from "react";
import { Typography, Descriptions, Card, Tag, Tabs, Alert, Collapse, Radio, Space, Spin } from "antd";
import moment from "moment";
import { LogEntry } from "../columns";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import GuardrailViewer from "../GuardrailViewer/GuardrailViewer";
import CompliancePanel from "../GuardrailViewer/CompliancePanel";
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

      {/* Metrics */}
      <MetricsSection logEntry={logEntry} metadata={metadata} />

      {/* Cost Breakdown */}
      <CostBreakdownViewer costBreakdown={metadata?.cost_breakdown} totalSpend={logEntry.spend || 0} />

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

  return (
    <div className="bg-white rounded-lg shadow w-full max-w-full overflow-hidden mb-6">
      <Card title="Metrics" size="small" bordered={false} style={{ marginBottom: 0 }}>
        <Descriptions column={2} size="small">
          <Descriptions.Item label="Tokens">
            <TokenFlow
              prompt={logEntry.prompt_tokens}
              completion={logEntry.completion_tokens}
              total={logEntry.total_tokens}
            />
          </Descriptions.Item>
          <Descriptions.Item label="Cost">${formatNumberWithCommas(logEntry.spend || 0, 8)}</Descriptions.Item>
          <Descriptions.Item label="Duration">{logEntry.duration?.toFixed(3)} s</Descriptions.Item>

          {hasCacheActivity && (
            <>
              <Descriptions.Item label="Cache Hit">
                <Tag color={logEntry.cache_hit ? "green" : "default"}>{logEntry.cache_hit || "None"}</Tag>
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

  const totalSpend = logEntry.spend || 0;
  const promptTokens = logEntry.prompt_tokens || 0;
  const completionTokens = logEntry.completion_tokens || 0;
  const totalTokens = promptTokens + completionTokens;
  const inputCost = totalTokens > 0 ? (totalSpend * promptTokens) / totalTokens : 0;
  const outputCost = totalTokens > 0 ? (totalSpend * completionTokens) / totalTokens : 0;

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
