import { useState } from "react";
import { Drawer, Typography, Descriptions, Card, Tag, Tabs, Alert, Collapse, Radio, Space } from "antd";
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
import { useKeyboardNavigation } from "./useKeyboardNavigation";
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
  DRAWER_WIDTH,
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
          <MetadataSection metadata={logEntry.metadata} />
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
  return (
    <Space size={SPACING_MEDIUM}>
      <span>{label}</span>
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
  getRawRequest: () => any;
  getFormattedResponse: () => any;
  logEntry: LogEntry;
}

function RequestResponseSection({
  hasResponse,
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
            label: (
              <div 
                style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}
                onClick={(e) => {
                  // Only prevent if clicking on the Radio.Group area
                  const target = e.target as HTMLElement;
                  if (target.closest('.ant-radio-group')) {
                    e.stopPropagation();
                  }
                }}
              >
                <h3 className="text-lg font-medium text-gray-900" style={{ margin: 0 }}>Request & Response</h3>
                {/* View Mode Toggle - In the header */}
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
                        disabled={activeTab === TAB_RESPONSE && !hasResponse}
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

