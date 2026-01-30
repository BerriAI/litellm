import { Drawer, Typography, Button, Descriptions, Card, Tag, Tooltip, Tabs, message } from "antd";
import { CloseOutlined, CopyOutlined } from "@ant-design/icons";
import { Row } from "@tanstack/react-table";
import { LogEntry } from "./columns";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { truncateString } from "@/utils/textUtils";
import GuardrailViewer from "./GuardrailViewer/GuardrailViewer";
import { CostBreakdownViewer } from "./CostBreakdownViewer";
import { ConfigInfoMessage } from "./ConfigInfoMessage";
import { RequestResponsePanel } from "./RequestResponsePanel";
import { VectorStoreViewer } from "./VectorStoreViewer";
import { ErrorViewer } from "./ErrorViewer";
import { JsonView, defaultStyles } from "react-json-view-lite";
import "react-json-view-lite/dist/index.css";

const { Title, Text } = Typography;

interface LogDetailsDrawerProps {
  open: boolean;
  onClose: () => void;
  logEntry: LogEntry | null;
  onOpenSettings?: () => void;
}

export function LogDetailsDrawer({ open, onClose, logEntry, onOpenSettings }: LogDetailsDrawerProps) {
  if (!logEntry) return null;

  // Helper function to clean metadata by removing specific fields
  const formatData = (input: any) => {
    if (typeof input === "string") {
      try {
        return JSON.parse(input);
      } catch {
        return input;
      }
    }
    return input;
  };

  // Helper function to get raw request
  const getRawRequest = () => {
    // First check if proxy_server_request exists in metadata
    if (logEntry?.proxy_server_request) {
      return formatData(logEntry.proxy_server_request);
    }
    // Fall back to messages if proxy_server_request is empty
    return formatData(logEntry.messages);
  };

  // Extract error information from metadata if available
  const metadata = logEntry.metadata || {};
  const hasError = metadata.status === "failure";
  const errorInfo = hasError ? metadata.error_information : null;

  // Check if request/response data is missing
  const hasMessages =
    logEntry.messages &&
    (Array.isArray(logEntry.messages)
      ? logEntry.messages.length > 0
      : Object.keys(logEntry.messages).length > 0);
  const hasResponse = logEntry.response && Object.keys(formatData(logEntry.response)).length > 0;
  const missingData = !hasMessages && !hasResponse;

  // Format the response with error details if present
  const formattedResponse = () => {
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

  // Extract vector store request metadata if available
  const hasVectorStoreData =
    metadata.vector_store_request_metadata &&
    Array.isArray(metadata.vector_store_request_metadata) &&
    metadata.vector_store_request_metadata.length > 0;

  // Extract guardrail information from metadata if available
  const guardrailInfo = logEntry.metadata?.guardrail_information;
  const guardrailEntries = Array.isArray(guardrailInfo) ? guardrailInfo : guardrailInfo ? [guardrailInfo] : [];
  const hasGuardrailData = guardrailEntries.length > 0;

  // Calculate total masked entities if guardrail data exists
  const totalMaskedEntities = guardrailEntries.reduce((sum, entry) => {
    const maskedCounts = entry?.masked_entity_count;
    if (!maskedCounts) {
      return sum;
    }
    return (
      sum +
      Object.values(maskedCounts).reduce<number>((acc, count) => (typeof count === "number" ? acc + count : acc), 0)
    );
  }, 0);

  const primaryGuardrailLabel =
    guardrailEntries.length === 1
      ? guardrailEntries[0]?.guardrail_name ?? "-"
      : guardrailEntries.length > 1
        ? `${guardrailEntries.length} guardrails`
        : "-";

  const handleCopyRequestId = () => {
    navigator.clipboard.writeText(logEntry.request_id);
    message.success("Request ID copied to clipboard");
  };

  const copyToClipboard = async (text: string, label: string) => {
    try {
      // Try modern clipboard API first
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
        message.success(`${label} copied to clipboard`);
        return true;
      } else {
        // Fallback for non-secure contexts
        const textArea = document.createElement("textarea");
        textArea.value = text;
        textArea.style.position = "fixed";
        textArea.style.opacity = "0";
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();

        const successful = document.execCommand("copy");
        document.body.removeChild(textArea);

        if (!successful) {
          throw new Error("execCommand failed");
        }
        message.success(`${label} copied to clipboard`);
        return true;
      }
    } catch (error) {
      console.error("Copy failed:", error);
      message.error(`Failed to copy ${label}`);
      return false;
    }
  };

  return (
    <Drawer
      title={null}
      placement="right"
      onClose={onClose}
      open={open}
      width="60%"
      closable={false}
      mask={true}
      maskClosable={true}
      styles={{
        body: { padding: 0, overflow: "hidden" },
        header: { display: "none" },
      }}
    >
      {/* Custom Header with Request ID prominently displayed */}
      <div
        style={{
          padding: "16px 24px",
          borderBottom: "1px solid #f0f0f0",
          backgroundColor: "#fff",
          position: "sticky",
          top: 0,
          zIndex: 10,
        }}
      >
        {/* Request ID at top - like Langfuse trace ID */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "8px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", flex: 1, minWidth: 0 }}>
            <Tooltip title={logEntry.request_id}>
              <Text
                strong
                style={{
                  fontSize: "16px",
                  fontFamily: "monospace",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
              >
                {logEntry.request_id}
              </Text>
            </Tooltip>
            <Tooltip title="Copy Request ID">
              <Button type="text" size="small" icon={<CopyOutlined />} onClick={handleCopyRequestId} />
            </Tooltip>
          </div>
          <Button type="text" icon={<CloseOutlined />} onClick={onClose} />
        </div>

        {/* Status and timestamp row */}
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <Tag color={metadata.status === "failure" ? "error" : "success"}>
            {metadata.status === "failure" ? "Failure" : "Success"}
          </Tag>
          <Text type="secondary" style={{ fontSize: "13px" }}>
            {logEntry.startTime}
          </Text>
        </div>
      </div>

      {/* Scrollable content area */}
      <div style={{ height: "calc(100vh - 100px)", overflowY: "auto", padding: "24px" }}>
        {/* Request Details Section */}
        <Card title="Request Details" size="small" style={{ marginBottom: "16px" }}>
          <Descriptions column={2} size="small">
            <Descriptions.Item label="Model">{logEntry.model}</Descriptions.Item>
            <Descriptions.Item label="Provider">{logEntry.custom_llm_provider || "-"}</Descriptions.Item>
            <Descriptions.Item label="Call Type">{logEntry.call_type}</Descriptions.Item>
            <Descriptions.Item label="Model ID">{logEntry.model_id}</Descriptions.Item>
            <Descriptions.Item label="API Base">
              <Tooltip title={logEntry.api_base || "-"}>
                <span
                  style={{
                    maxWidth: "200px",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                    display: "inline-block",
                  }}
                >
                  {logEntry.api_base || "-"}
                </span>
              </Tooltip>
            </Descriptions.Item>
            {logEntry.requester_ip_address && (
              <Descriptions.Item label="IP Address">{logEntry.requester_ip_address}</Descriptions.Item>
            )}
            {hasGuardrailData && (
              <Descriptions.Item label="Guardrail">
                <span>{primaryGuardrailLabel}</span>
                {totalMaskedEntities > 0 && (
                  <Tag color="blue" style={{ marginLeft: "8px" }}>
                    {totalMaskedEntities} masked
                  </Tag>
                )}
              </Descriptions.Item>
            )}
          </Descriptions>
        </Card>

        {/* Metrics Section */}
        <Card title="Metrics" size="small" style={{ marginBottom: "16px" }}>
          <Descriptions column={2} size="small">
            <Descriptions.Item label="Tokens">
              {logEntry.total_tokens} ({logEntry.prompt_tokens} prompt + {logEntry.completion_tokens} completion)
            </Descriptions.Item>
            <Descriptions.Item label="Cost">${formatNumberWithCommas(logEntry.spend || 0, 6)}</Descriptions.Item>
            <Descriptions.Item label="Duration">{logEntry.duration} s</Descriptions.Item>
            <Descriptions.Item label="Cache Hit">{logEntry.cache_hit}</Descriptions.Item>
            <Descriptions.Item label="Cache Read Tokens">
              {formatNumberWithCommas(metadata?.additional_usage_values?.cache_read_input_tokens || 0)}
            </Descriptions.Item>
            <Descriptions.Item label="Cache Creation Tokens">
              {formatNumberWithCommas(metadata?.additional_usage_values?.cache_creation_input_tokens || 0)}
            </Descriptions.Item>
            <Descriptions.Item label="Start Time">{logEntry.startTime}</Descriptions.Item>
            <Descriptions.Item label="End Time">{logEntry.endTime}</Descriptions.Item>
            {metadata?.litellm_overhead_time_ms !== undefined && (
              <Descriptions.Item label="LiteLLM Overhead">{metadata.litellm_overhead_time_ms} ms</Descriptions.Item>
            )}
          </Descriptions>
        </Card>

        {/* Cost Breakdown - Show if cost breakdown data is available */}
        <CostBreakdownViewer costBreakdown={metadata?.cost_breakdown} totalSpend={logEntry.spend || 0} />

        {/* Configuration Info Message - Show when data is missing */}
        <ConfigInfoMessage show={missingData} onOpenSettings={onOpenSettings} />

        {/* Request/Response JSON - Using Tabs */}
        <Card title="Request & Response" size="small" style={{ marginBottom: "16px" }}>
          <Tabs
            defaultActiveKey="request"
            items={[
              {
                key: "request",
                label: "Request",
                children: (
                  <div style={{ position: "relative" }}>
                    <Button
                      type="text"
                      size="small"
                      icon={<CopyOutlined />}
                      style={{ position: "absolute", top: "8px", right: "8px", zIndex: 1 }}
                      onClick={() => copyToClipboard(JSON.stringify(getRawRequest(), null, 2), "Request")}
                    >
                      Copy
                    </Button>
                    <div
                      style={{
                        maxHeight: "500px",
                        overflowY: "auto",
                        backgroundColor: "#fafafa",
                        padding: "12px",
                        borderRadius: "4px",
                      }}
                    >
                      <div className="[&_[role='tree']]:bg-white [&_[role='tree']]:text-slate-900">
                        <JsonView data={getRawRequest()} style={defaultStyles} clickToExpandNode={true} />
                      </div>
                    </div>
                  </div>
                ),
              },
              {
                key: "response",
                label: "Response",
                children: (
                  <div style={{ position: "relative" }}>
                    <Button
                      type="text"
                      size="small"
                      icon={<CopyOutlined />}
                      style={{ position: "absolute", top: "8px", right: "8px", zIndex: 1 }}
                      onClick={() => copyToClipboard(JSON.stringify(formattedResponse(), null, 2), "Response")}
                      disabled={!hasResponse}
                    >
                      Copy
                    </Button>
                    <div
                      style={{
                        maxHeight: "500px",
                        overflowY: "auto",
                        backgroundColor: "#fafafa",
                        padding: "12px",
                        borderRadius: "4px",
                      }}
                    >
                      {hasResponse ? (
                        <div className="[&_[role='tree']]:bg-white [&_[role='tree']]:text-slate-900">
                          <JsonView data={formattedResponse()} style={defaultStyles} clickToExpandNode />
                        </div>
                      ) : (
                        <div style={{ textAlign: "center", padding: "20px", color: "#999", fontStyle: "italic" }}>
                          Response data not available
                        </div>
                      )}
                    </div>
                  </div>
                ),
              },
            ]}
          />
        </Card>

        {/* Guardrail Data - Show only if present */}
        {hasGuardrailData && (
          <div style={{ marginBottom: "16px" }}>
            <GuardrailViewer data={guardrailInfo} />
          </div>
        )}

        {/* Vector Store Request Data - Show only if present */}
        {hasVectorStoreData && (
          <div style={{ marginBottom: "16px" }}>
            <VectorStoreViewer data={metadata.vector_store_request_metadata} />
          </div>
        )}

        {/* Error Card - Only show for failures */}
        {hasError && errorInfo && (
          <div style={{ marginBottom: "16px" }}>
            <ErrorViewer errorInfo={errorInfo} />
          </div>
        )}

        {/* Tags Card - Only show if there are tags */}
        {logEntry.request_tags && Object.keys(logEntry.request_tags).length > 0 && (
          <Card title="Request Tags" size="small" style={{ marginBottom: "16px" }}>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
              {Object.entries(logEntry.request_tags).map(([key, value]) => (
                <Tag key={key}>
                  {key}: {String(value)}
                </Tag>
              ))}
            </div>
          </Card>
        )}

        {/* Metadata Card - Only show if there's metadata */}
        {logEntry.metadata && Object.keys(logEntry.metadata).length > 0 && (
          <Card
            title="Metadata"
            size="small"
            style={{ marginBottom: "16px" }}
            extra={
              <Button
                type="text"
                size="small"
                icon={<CopyOutlined />}
                onClick={() => copyToClipboard(JSON.stringify(logEntry.metadata, null, 2), "Metadata")}
              >
                Copy
              </Button>
            }
          >
            <pre
              style={{
                maxHeight: "300px",
                overflowY: "auto",
                fontSize: "12px",
                fontFamily: "monospace",
                whiteSpace: "pre-wrap",
                wordBreak: "break-all",
                margin: 0,
              }}
            >
              {JSON.stringify(logEntry.metadata, null, 2)}
            </pre>
          </Card>
        )}
      </div>
    </Drawer>
  );
}
