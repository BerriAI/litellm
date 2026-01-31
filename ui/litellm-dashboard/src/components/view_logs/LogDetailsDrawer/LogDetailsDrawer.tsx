import { useState } from "react";
import { Drawer, Typography, Button, Descriptions, Card, Tag, Tabs, Alert, message, Collapse, Radio } from "antd";
import { CopyOutlined } from "@ant-design/icons";
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
import { DrawerHeader } from "./DrawerHeader";
import { copyToClipboard } from "./clipboardUtils";
import { useKeyboardNavigation } from "./useKeyboardNavigation";
import {
  DRAWER_WIDTH,
  DRAWER_CONTENT_PADDING,
  API_BASE_MAX_WIDTH,
  METADATA_MAX_HEIGHT,
  TAB_REQUEST,
  TAB_RESPONSE,
  FONT_SIZE_SMALL,
  FONT_FAMILY_MONO,
  SPACING_XLARGE,
  MESSAGE_REQUEST_ID_COPIED,
} from "./constants";
import { ToolsSection } from "../ToolsSection";
import { PrettyMessagesView } from "./PrettyMessagesView";

const { Text } = Typography;

export interface LogDetailsDrawerProps {
  open: boolean;
  onClose: () => void;
  logEntry: LogEntry | null;
  onOpenSettings?: () => void;
  allLogs?: LogEntry[];
  onSelectLog?: (log: LogEntry) => void;
}

/**
 * Right-side drawer panel for displaying detailed log information.
 * Features:
 * - Request ID prominently displayed with copy functionality
 * - Keyboard navigation (J/K for next/prev, Escape to close)
 * - Formatted and JSON view toggle for request/response
 * - Smart display of cache fields (hidden when zero)
 * - Error alerts for failed requests
 * - Collapsible sections for guardrails, vector store, metadata
 */
export function LogDetailsDrawer({
  open,
  onClose,
  logEntry,
  onOpenSettings,
  allLogs = [],
  onSelectLog,
}: LogDetailsDrawerProps) {
  const [activeTab, setActiveTab] = useState<typeof TAB_REQUEST | typeof TAB_RESPONSE>(TAB_REQUEST);

  // Keyboard navigation
  const { selectNextLog, selectPreviousLog } = useKeyboardNavigation({
    isOpen: open,
    currentLog: logEntry,
    allLogs,
    onClose,
    onSelectLog,
  });

  if (!logEntry) return null;

  const metadata = logEntry.metadata || {};
  const hasError = metadata.status === "failure";
  const errorInfo = hasError ? metadata.error_information : null;

  // Check if request/response data is present
  const hasMessages = checkHasMessages(logEntry.messages);
  const hasResponse = checkHasResponse(logEntry.response);
  const missingData = !hasMessages && !hasResponse;

  // Guardrail data
  const guardrailInfo = metadata?.guardrail_information;
  const guardrailEntries = normalizeGuardrailEntries(guardrailInfo);
  const hasGuardrailData = guardrailEntries.length > 0;
  const totalMaskedEntities = calculateTotalMaskedEntities(guardrailEntries);
  const primaryGuardrailLabel = getGuardrailLabel(guardrailEntries);

  // Vector store data
  const hasVectorStoreData = checkHasVectorStoreData(metadata);

  // Status display values
  const statusLabel = metadata.status === "failure" ? "Failure" : "Success";
  const statusColor = metadata.status === "failure" ? ("error" as const) : ("success" as const);
  const environment = metadata?.user_api_key_team_alias || "default";

  const handleCopyRequestId = () => {
    navigator.clipboard.writeText(logEntry.request_id);
    message.success(MESSAGE_REQUEST_ID_COPIED);
  };

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
    <Drawer
      title={null}
      placement="right"
      onClose={onClose}
      open={open}
      width={DRAWER_WIDTH}
      closable={false}
      mask={true}
      maskClosable={true}
      styles={{
        body: { padding: 0, overflow: "hidden" },
        header: { display: "none" },
      }}
    >
      <DrawerHeader
        log={logEntry}
        onClose={onClose}
        onCopyRequestId={handleCopyRequestId}
        onPrevious={selectPreviousLog}
        onNext={selectNextLog}
        statusLabel={statusLabel}
        statusColor={statusColor}
        environment={environment}
      />

      <div style={{ height: "calc(100vh - 100px)", overflowY: "auto", padding: `${DRAWER_CONTENT_PADDING} ${DRAWER_CONTENT_PADDING} 0` }}>
        {/* Error Alert - Show prominently at top for failures */}
        {hasError && errorInfo && (
          <Alert
            type="error"
            showIcon
            message="Request Failed"
            description={<ErrorDescription errorInfo={errorInfo} />}
            className="mb-6"
          />
        )}

        {/* Tags - Only show if present */}
        {logEntry.request_tags && Object.keys(logEntry.request_tags).length > 0 && (
          <TagsSection tags={logEntry.request_tags} />
        )}

        {/* Request Details Section */}
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

        {/* Metrics Section */}
        <MetricsSection logEntry={logEntry} metadata={metadata} />

        {/* Cost Breakdown - Show if cost breakdown data is available */}
        <CostBreakdownViewer costBreakdown={metadata?.cost_breakdown} totalSpend={logEntry.spend || 0} />

        {/* Tools Section - Show if tools are present in request */}
        <ToolsSection log={logEntry} />

        {/* Configuration Info Message - Show when data is missing */}
        {missingData && (
          <div className="mb-6">
            <ConfigInfoMessage show={missingData} onOpenSettings={onOpenSettings} />
          </div>
        )}

        {/* Request/Response JSON - Collapsible */}
        <RequestResponseSection
          hasResponse={hasResponse}
          onCopy={(data, label) => copyToClipboard(JSON.stringify(data, null, 2), label)}
          getRawRequest={getRawRequest}
          getFormattedResponse={getFormattedResponse}
          logEntry={logEntry}
        />

        {/* Guardrail Data - Show only if present */}
        {hasGuardrailData && <GuardrailViewer data={guardrailInfo} />}

        {/* Vector Store Request Data - Show only if present */}
        {hasVectorStoreData && <VectorStoreViewer data={metadata.vector_store_request_metadata} />}

        {/* Metadata Card - Only show if there's metadata */}
        {logEntry.metadata && Object.keys(logEntry.metadata).length > 0 && (
          <MetadataSection metadata={logEntry.metadata} onCopy={(data) => copyToClipboard(data, "Metadata")} />
        )}
        
        {/* Bottom spacing for scroll area */}
        <div style={{ height: DRAWER_CONTENT_PADDING }} />
      </div>
    </Drawer>
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
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        {Object.entries(tags).map(([key, value]) => (
          <Tag key={key}>
            {key}: {String(value)}
          </Tag>
        ))}
      </div>
    </div>
  );
}

function GuardrailLabel({ label, maskedCount }: { label: string; maskedCount: number }) {
  return (
    <>
      <span>{label}</span>
      {maskedCount > 0 && (
        <Tag color="blue" style={{ marginLeft: 8 }}>
          {maskedCount} masked
        </Tag>
      )}
    </>
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

        {/* Only show cache fields if there's cache activity */}
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
  onCopy: (data: any, label: string) => void;
  getRawRequest: () => any;
  getFormattedResponse: () => any;
  logEntry: LogEntry;
}

function RequestResponseSection({
  hasResponse,
  onCopy,
  getRawRequest,
  getFormattedResponse,
  logEntry,
}: RequestResponseSectionProps) {
  const [activeTab, setActiveTab] = useState<typeof TAB_REQUEST | typeof TAB_RESPONSE>(TAB_REQUEST);
  const [viewMode, setViewMode] = useState<'pretty' | 'json'>('pretty');

  const handleCopy = () => {
    const data = activeTab === TAB_REQUEST ? getRawRequest() : getFormattedResponse();
    const label = activeTab === TAB_REQUEST ? "Request" : "Response";
    onCopy(data, label);
  };

  // Calculate input and output costs
  // Assume average cost if not explicitly provided
  const totalSpend = logEntry.spend || 0;
  const promptTokens = logEntry.prompt_tokens || 0;
  const completionTokens = logEntry.completion_tokens || 0;
  const totalTokens = promptTokens + completionTokens;
  
  // Estimate input/output costs proportionally if not available
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
            label: <h3 className="text-lg font-medium text-gray-900">Request & Response</h3>,
            children: (
              <div style={{ padding: "0 24px" }}>
                {/* View Mode Toggle - Top Right */}
                <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 16 }}>
                  <Radio.Group
                    size="small"
                    value={viewMode}
                    onChange={(e) => setViewMode(e.target.value)}
                  >
                    <Radio.Button value="pretty">Pretty</Radio.Button>
                    <Radio.Button value="json">JSON</Radio.Button>
                  </Radio.Group>
                </div>

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
                      <Button
                        type="text"
                        size="small"
                        icon={<CopyOutlined />}
                        onClick={handleCopy}
                        disabled={activeTab === TAB_RESPONSE && !hasResponse}
                      >
                        Copy
                      </Button>
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
                            {hasResponse ? (
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

function MetadataSection({ metadata, onCopy }: { metadata: Record<string, any>; onCopy: (data: string) => void }) {
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
                  <Button
                    type="text"
                    size="small"
                    icon={<CopyOutlined />}
                    onClick={() => onCopy(JSON.stringify(metadata, null, 2))}
                  >
                    Copy
                  </Button>
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

// ============================================================================
// Helper Functions
// ============================================================================

function formatData(input: any) {
  if (typeof input === "string") {
    try {
      return JSON.parse(input);
    } catch {
      return input;
    }
  }
  return input;
}

function checkHasMessages(messages: any): boolean {
  if (!messages) return false;
  if (Array.isArray(messages)) return messages.length > 0;
  if (typeof messages === "object") return Object.keys(messages).length > 0;
  return false;
}

function checkHasResponse(response: any): boolean {
  if (!response) return false;
  return Object.keys(formatData(response)).length > 0;
}

function normalizeGuardrailEntries(guardrailInfo: any): any[] {
  if (Array.isArray(guardrailInfo)) return guardrailInfo;
  if (guardrailInfo) return [guardrailInfo];
  return [];
}

function calculateTotalMaskedEntities(entries: any[]): number {
  return entries.reduce((sum, entry) => {
    const maskedCounts = entry?.masked_entity_count;
    if (!maskedCounts) return sum;
    return (
      sum +
      Object.values(maskedCounts).reduce<number>((acc, count) => (typeof count === "number" ? acc + count : acc), 0)
    );
  }, 0);
}

function getGuardrailLabel(entries: any[]): string {
  if (entries.length === 0) return "-";
  if (entries.length === 1) return entries[0]?.guardrail_name ?? "-";
  return `${entries.length} guardrails`;
}

function checkHasVectorStoreData(metadata: Record<string, any>): boolean {
  return (
    metadata.vector_store_request_metadata &&
    Array.isArray(metadata.vector_store_request_metadata) &&
    metadata.vector_store_request_metadata.length > 0
  );
}
