import { useState } from "react";
import moment from "moment";
import { Copy as CopyIcon, LoaderCircle } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
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
} from "./constants";
import { ToolsSection } from "../ToolsSection";
import { PrettyMessagesView } from "./PrettyMessagesView";

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
        <Alert variant="destructive" className="mb-6">
          <AlertTitle>Request Failed</AlertTitle>
          <AlertDescription>
            <ErrorDescription errorInfo={errorInfo} />
          </AlertDescription>
        </Alert>
      )}

      {/* Tags */}
      {logEntry.request_tags && Object.keys(logEntry.request_tags).length > 0 && (
        <TagsSection tags={logEntry.request_tags} />
      )}

      {/* Request Details */}
      <div className="bg-background rounded-lg shadow w-full max-w-full overflow-hidden mb-6">
        <Card className="p-0 border-0 shadow-none rounded-none">
          <SectionTitle>Request Details</SectionTitle>
          <DescriptionGrid columns={2}>
            <DescriptionItem label="Model">{logEntry.model}</DescriptionItem>
            <DescriptionItem label="Provider">{logEntry.custom_llm_provider || "-"}</DescriptionItem>
            <DescriptionItem label="Call Type">{logEntry.call_type}</DescriptionItem>
            <DescriptionItem label="Model ID">
              <TruncatedValue value={logEntry.model_id} />
            </DescriptionItem>
            <DescriptionItem label="API Base">
              <TruncatedValue value={logEntry.api_base} maxWidth={API_BASE_MAX_WIDTH} />
            </DescriptionItem>
            {logEntry.requester_ip_address && (
              <DescriptionItem label="IP Address">{logEntry.requester_ip_address}</DescriptionItem>
            )}
            {hasGuardrailData && (
              <DescriptionItem label="Guardrail">
                <GuardrailLabel label={primaryGuardrailLabel} maskedCount={totalMaskedEntities} />
              </DescriptionItem>
            )}
          </DescriptionGrid>
        </Card>
      </div>

      {/* Metrics */}
      <MetricsSection logEntry={logEntry} metadata={metadata} />

      {/* Cost Breakdown */}
      <CostBreakdownViewer
        costBreakdown={metadata?.cost_breakdown}
        totalSpend={logEntry.spend ?? 0}
        promptTokens={logEntry.prompt_tokens}
        completionTokens={logEntry.completion_tokens}
        cacheHit={logEntry.cache_hit}
        rawInputTokens={metadata?.additional_usage_values?.prompt_tokens_details?.text_tokens}
        cacheReadTokens={metadata?.additional_usage_values?.cache_read_input_tokens}
        cacheCreationTokens={metadata?.additional_usage_values?.cache_creation_input_tokens}
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
        <div className="bg-background rounded-lg shadow w-full max-w-full overflow-hidden mb-6 p-8 text-center">
          <LoaderCircle className="animate-spin mx-auto text-muted-foreground" size={20} />
          <div className="mt-2 text-muted-foreground">Loading request &amp; response data...</div>
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

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <div className="px-4 py-2 border-b text-sm font-semibold">
      {children}
    </div>
  );
}

/** Replacement for antd `Descriptions` with `column={n}` layout — renders a
 *  semantic `<dl>` as a grid of label/value pairs. */
function DescriptionGrid({
  columns = 2,
  children,
}: {
  columns?: number;
  children: React.ReactNode;
}) {
  const gridCols = columns === 2 ? "md:grid-cols-2" : columns === 3 ? "md:grid-cols-3" : "md:grid-cols-1";
  return (
    <dl className={`grid grid-cols-1 ${gridCols} gap-x-6 gap-y-2 p-4 text-sm`}>
      {children}
    </dl>
  );
}

function DescriptionItem({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="ant-descriptions-item flex gap-2" role="group" aria-label={label}>
      <dt className="text-muted-foreground font-medium min-w-[8rem]">{label}</dt>
      <dd className="text-foreground">{children}</dd>
    </div>
  );
}

function ErrorDescription({ errorInfo }: { errorInfo: any }) {
  return (
    <div>
      {errorInfo.error_code && (
        <div>
          <strong>Error Code:</strong> {errorInfo.error_code}
        </div>
      )}
      {errorInfo.error_message && (
        <div>
          <strong>Message:</strong> {errorInfo.error_message}
        </div>
      )}
    </div>
  );
}

function TagsSection({ tags }: { tags: Record<string, any> }) {
  return (
    <div className="bg-background rounded-lg shadow w-full max-w-full overflow-hidden p-4 mb-6">
      <div className="font-semibold text-base mb-2">Tags</div>
      <div className="flex flex-wrap gap-2">
        {Object.entries(tags).map(([key, value]) => (
          <Badge key={key} variant="secondary">
            {key}: {String(value)}
          </Badge>
        ))}
      </div>
    </div>
  );
}

function GuardrailLabel({ label, maskedCount }: { label: string; maskedCount: number }) {
  const handleClick = () => {
    const el = document.getElementById("guardrail-section");
    if (el) el.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <span className="inline-flex items-center gap-2">
      <a onClick={handleClick} className="cursor-pointer hover:underline">{label}</a>
      {maskedCount > 0 && (
        <Badge className="bg-blue-100 text-blue-700 hover:bg-blue-100">
          {maskedCount} masked
        </Badge>
      )}
    </span>
  );
}

/**
 * Uncached input token count (billable non-cache prompt text), aligned with Cost Breakdown "Input".
 * Same sources as CostBreakdownViewer rawInputTokens.
 */
function getUncachedInputTextTokens(metadata: Record<string, any>): number | undefined {
  const raw =
    metadata?.additional_usage_values?.prompt_tokens_details?.text_tokens ??
    metadata?.usage_object?.prompt_tokens_details?.text_tokens;
  if (raw === undefined || raw === null) return undefined;
  const n = Number(raw);
  return Number.isFinite(n) ? n : undefined;
}

function MetricsSection({ logEntry, metadata }: { logEntry: LogEntry; metadata: Record<string, any> }) {
  const completionStartTime = logEntry.completionStartTime;
  const ttftMs =
    completionStartTime && completionStartTime !== logEntry.endTime
      ? new Date(completionStartTime).getTime() - new Date(logEntry.startTime).getTime()
      : null;

  const hasCacheActivity =
    logEntry.cache_hit ||
    (metadata?.additional_usage_values?.cache_read_input_tokens &&
      metadata.additional_usage_values.cache_read_input_tokens > 0);

  const cacheHitValue = String(logEntry.cache_hit ?? "None");
  const cacheHitBadgeClass =
    cacheHitValue.toLowerCase() === "true"
      ? "bg-green-100 text-green-700"
      : cacheHitValue.toLowerCase() === "false"
        ? "bg-red-100 text-red-700"
        : "bg-muted text-foreground";

  const uncachedInputTokens = getUncachedInputTextTokens(metadata);
  const showAnthropicMessagesInputOutput =
    logEntry.call_type === "anthropic_messages" && uncachedInputTokens !== undefined;

  return (
    <div className="bg-background rounded-lg shadow w-full max-w-full overflow-hidden mb-6">
      <Card className="p-0 border-0 shadow-none rounded-none">
        <SectionTitle>Metrics</SectionTitle>
        <DescriptionGrid columns={2}>
          {showAnthropicMessagesInputOutput ? (
            <>
              <DescriptionItem label="Input Tokens">
                {formatNumberWithCommas(uncachedInputTokens)}
              </DescriptionItem>
              <DescriptionItem label="Output Tokens">
                {formatNumberWithCommas(logEntry.completion_tokens)}
              </DescriptionItem>
            </>
          ) : (
            <DescriptionItem label="Tokens">
              <TokenFlow
                prompt={logEntry.prompt_tokens}
                completion={logEntry.completion_tokens}
                total={logEntry.total_tokens}
              />
            </DescriptionItem>
          )}
          <DescriptionItem label="Cost">${formatNumberWithCommas(logEntry.spend || 0, 8)}</DescriptionItem>
          <DescriptionItem label="Duration">{logEntry.request_duration_ms != null ? (logEntry.request_duration_ms / 1000).toFixed(3) : "-"} s</DescriptionItem>
          {ttftMs != null && ttftMs > 0 && (
            <DescriptionItem label="Time to First Token">{(ttftMs / 1000).toFixed(3)} s</DescriptionItem>
          )}

          {hasCacheActivity && (
            <>
              <DescriptionItem label="Cache Hit">
                <Badge className={cacheHitBadgeClass}>{cacheHitValue}</Badge>
              </DescriptionItem>
              {metadata?.additional_usage_values?.cache_read_input_tokens > 0 && (
                <DescriptionItem label="Cache Read Tokens">
                  {formatNumberWithCommas(metadata.additional_usage_values.cache_read_input_tokens)}
                </DescriptionItem>
              )}
              {metadata?.additional_usage_values?.cache_creation_input_tokens > 0 && (
                <DescriptionItem label="Cache Creation Tokens">
                  {formatNumberWithCommas(metadata.additional_usage_values.cache_creation_input_tokens)}
                </DescriptionItem>
              )}
            </>
          )}

          {metadata?.litellm_overhead_time_ms !== undefined && metadata.litellm_overhead_time_ms !== null && (
            <DescriptionItem label="LiteLLM Overhead">
              {metadata.litellm_overhead_time_ms.toFixed(2)} ms
            </DescriptionItem>
          )}

          <DescriptionItem label="Retries">
            {metadata?.attempted_retries !== undefined && metadata?.attempted_retries !== null
              ? metadata.attempted_retries > 0
                ? <>{metadata.attempted_retries}{metadata.max_retries !== undefined && metadata.max_retries !== null ? ` / ${metadata.max_retries}` : ''}</>
                : <Badge className="bg-green-100 text-green-700 hover:bg-green-100">None</Badge>
              : "-"}
          </DescriptionItem>

          <DescriptionItem label="Start Time">
            {moment(logEntry.startTime).format("YYYY-MM-DDTHH:mm:ss.SSS[Z]")}
          </DescriptionItem>
          <DescriptionItem label="End Time">
            {moment(logEntry.endTime).format("YYYY-MM-DDTHH:mm:ss.SSS[Z]")}
          </DescriptionItem>
        </DescriptionGrid>
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

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(getCopyText());
    } catch {
      /* noop */
    }
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
    <div className="bg-background rounded-lg shadow w-full max-w-full overflow-hidden mb-6">
      <Accordion type="single" collapsible defaultValue="request-response" className="w-full">
        <AccordionItem value="request-response" className="border-b-0">
          <div className="flex items-center justify-between pr-4">
            <AccordionTrigger className="flex-1 px-4 py-3 hover:no-underline">
              <h3 className="text-lg font-medium text-foreground m-0">Request &amp; Response</h3>
            </AccordionTrigger>
            <div onClick={(e) => e.stopPropagation()}>
              <RadioGroup
                value={viewMode}
                onValueChange={(v) => setViewMode(v as 'pretty' | 'json')}
                className="flex items-center gap-2"
              >
                <div className="flex items-center gap-1">
                  <RadioGroupItem value="pretty" id="view-mode-pretty" />
                  <Label htmlFor="view-mode-pretty" className="cursor-pointer text-sm font-normal">Pretty</Label>
                </div>
                <div className="flex items-center gap-1">
                  <RadioGroupItem value="json" id="view-mode-json" />
                  <Label htmlFor="view-mode-json" className="cursor-pointer text-sm font-normal">JSON</Label>
                </div>
              </RadioGroup>
            </div>
          </div>
          <AccordionContent>
            <div className="px-4">
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
                <Tabs value={activeTab} onValueChange={(k) => setActiveTab(k as typeof TAB_REQUEST | typeof TAB_RESPONSE)}>
                  <div className="flex items-center justify-between">
                    <TabsList>
                      <TabsTrigger value={TAB_REQUEST}>Request</TabsTrigger>
                      <TabsTrigger value={TAB_RESPONSE}>Response</TabsTrigger>
                    </TabsList>
                    <button
                      type="button"
                      onClick={handleCopy}
                      disabled={activeTab === TAB_RESPONSE && !hasResponse && !hasError}
                      className="p-1 text-muted-foreground hover:text-foreground disabled:opacity-40"
                      aria-label="Copy JSON"
                      title="Copy JSON"
                    >
                      <CopyIcon size={14} />
                    </button>
                  </div>
                  <TabsContent value={TAB_REQUEST}>
                    <div style={{ paddingTop: SPACING_XLARGE, paddingBottom: SPACING_XLARGE }}>
                      <JsonViewer data={getRawRequest()} mode="formatted" />
                    </div>
                  </TabsContent>
                  <TabsContent value={TAB_RESPONSE}>
                    <div style={{ paddingTop: SPACING_XLARGE, paddingBottom: SPACING_XLARGE }}>
                      {hasResponse || hasError ? (
                        <JsonViewer data={getFormattedResponse()} mode="formatted" />
                      ) : (
                        <div className="text-center p-5 text-muted-foreground italic">
                          Response data not available
                        </div>
                      )}
                    </div>
                  </TabsContent>
                </Tabs>
              )}
            </div>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
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
    <div className="text-left mb-3">
      <div
        onClick={handleClick}
        className={`inline-flex items-center gap-1.5 py-1 px-3 rounded-2xl cursor-pointer text-xs font-medium border ${
          allPassed
            ? "bg-green-50 text-green-700 border-green-200"
            : "bg-red-50 text-red-700 border-red-200"
        }`}
      >
        {allPassed ? "\u2713" : "\u2717"} {guardrailEntries.length} guardrail{guardrailEntries.length !== 1 ? "s" : ""} evaluated
        <span className="text-[11px] opacity-70">{"\u2193"}</span>
      </div>
    </div>
  );
}

function MetadataSection({ metadata }: { metadata: Record<string, any> }) {
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(JSON.stringify(metadata, null, 2));
    } catch {
      /* noop */
    }
  };

  return (
    <div className="bg-background rounded-lg shadow w-full max-w-full overflow-hidden mb-6">
      <Accordion type="single" collapsible defaultValue="metadata" className="w-full">
        <AccordionItem value="metadata" className="border-b-0">
          <AccordionTrigger className="px-4 py-3 hover:no-underline">
            <h3 className="text-lg font-medium text-foreground m-0">Metadata</h3>
          </AccordionTrigger>
          <AccordionContent>
            <div className="px-4 pb-4">
              <div className="flex justify-end mb-2">
                <button
                  type="button"
                  onClick={handleCopy}
                  className="p-1 text-muted-foreground hover:text-foreground"
                  aria-label="Copy Metadata"
                  title="Copy Metadata"
                >
                  <CopyIcon size={14} />
                </button>
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
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </div>
  );
}
