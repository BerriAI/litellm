import { useState } from "react";
import { Check, ChevronDown, ChevronRight, CircleAlert, Copy, Info } from "lucide-react";
import moment from "moment";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { LogEntry } from "../columns";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { PROMPT_CACHE_CREATION_TOOLTIP, PROMPT_CACHE_READ_TOOLTIP } from "@/utils/promptCacheUsage";
import GuardrailViewer from "../GuardrailViewer/GuardrailViewer";
import EvalViewer from "../EvalViewer/EvalViewer";
import { CostBreakdownViewer } from "../CostBreakdownViewer";
import { ConfigInfoMessage } from "../ConfigInfoMessage";
import { VectorStoreViewer } from "../VectorStoreViewer";
import { TruncatedValue } from "./TruncatedValue";
import { TokenFlow } from "./TokenFlow";
import { JsonViewer } from "./JsonViewer";
import { RoutingDecisionCard, type RoutingDecision } from "./RoutingDecisionCard";
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
export function LogDetailContent({ logEntry, isLoadingDetails = false, accessToken }: LogDetailContentProps) {
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

  // LLM Judge data
  const evalInfo = metadata?.eval_information;
  const hasEvalData = evalInfo != null;

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
        <div
          role="alert"
          className="mb-6 flex items-start gap-2 rounded-lg border border-destructive/30 bg-destructive/5 p-3 text-sm"
        >
          <CircleAlert className="size-4 shrink-0 text-destructive" />
          <div>
            <div className="font-medium text-destructive">Request Failed</div>
            <ErrorDescription errorInfo={errorInfo} />
          </div>
        </div>
      )}

      {/* Tags */}
      {logEntry.request_tags && Object.keys(logEntry.request_tags).length > 0 && (
        <TagsSection tags={logEntry.request_tags} />
      )}

      {/* Request Details */}
      <div className="bg-card rounded-lg shadow-sm w-full max-w-full overflow-hidden mb-6">
        <Card size="sm" style={{ marginBottom: 0 }}>
          <CardHeader>
            <CardTitle>Request Details</CardTitle>
          </CardHeader>
          <CardContent>
            <DescriptionList>
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
            </DescriptionList>
          </CardContent>
        </Card>
      </div>

      {/* Routing */}
      <RoutingDecisionCard decision={metadata?.routing_decision as RoutingDecision | undefined} />

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
          <ConfigInfoMessage show={missingData} />
        </div>
      )}

      {/* Request/Response JSON */}
      {isLoadingDetails ? (
        <div className="bg-card rounded-lg shadow-sm w-full max-w-full overflow-hidden mb-6 p-8 text-center">
          <UiLoadingSpinner className="inline-block size-5" />
          <div style={{ marginTop: 8, color: "var(--color-muted-foreground)" }}>
            Loading request &amp; response data...
          </div>
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

      {/* LLM Judge Results */}
      {hasEvalData && <EvalViewer data={evalInfo} />}

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

function DescriptionList({ children }: { children: React.ReactNode }) {
  return <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">{children}</div>;
}

function DescriptionItem({ label, children }: { label: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="flex min-w-0 flex-wrap items-start gap-x-2 gap-y-0.5">
      <span className="shrink-0 text-muted-foreground after:content-[':']">{label}</span>
      <span className="min-w-0 break-words">{children}</span>
    </div>
  );
}

function CopyButton({
  getText,
  label,
  disabled = false,
}: {
  getText: () => string;
  label: string;
  disabled?: boolean;
}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(getText());
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {
      /* clipboard unavailable in non-secure contexts */
    }
  };

  return (
    <Button
      variant="ghost"
      size="icon-sm"
      onClick={handleCopy}
      disabled={disabled}
      aria-label={copied ? "Copied!" : label}
    >
      {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
    </Button>
  );
}

function ErrorDescription({ errorInfo }: { errorInfo: any }) {
  return (
    <div>
      {errorInfo.error_code && (
        <div>
          <span className="font-semibold">Error Code:</span> {errorInfo.error_code}
        </div>
      )}
      {errorInfo.error_message && (
        <div>
          <span className="font-semibold">Message:</span> {errorInfo.error_message}
        </div>
      )}
    </div>
  );
}

function TagsSection({ tags }: { tags: Record<string, any> }) {
  return (
    <div className="bg-card rounded-lg shadow-sm w-full max-w-full overflow-hidden p-4 mb-6">
      <span className="font-semibold" style={{ display: "block", marginBottom: 8, fontSize: 16 }}>
        Tags
      </span>
      <div className="flex flex-wrap items-center gap-2">
        {Object.entries(tags).map(([key, value]) => (
          <Badge key={key} variant="outline">
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
      <a onClick={handleClick} style={{ cursor: "pointer" }}>
        {label}
      </a>
      {maskedCount > 0 && <Badge variant="secondary">{maskedCount} masked</Badge>}
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

const RESPONSE_CACHE_TOOLTIP =
  "Whether this request was served from LiteLLM's response cache (e.g. Redis / in-memory), skipping the LLM provider call entirely. This is separate from provider prompt caching; a Miss here does not mean prompt caching failed.";
const RESPONSE_CACHE_DOCS_URL = "https://docs.litellm.ai/docs/proxy/caching";
const PROMPT_CACHE_DOCS_URL = "https://docs.litellm.ai/docs/completion/prompt_caching";

function MetricLabel({ label, tooltip, docsUrl }: { label: string; tooltip: string; docsUrl: string }) {
  return (
    <span className="inline-flex items-center gap-1">
      {label}
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger
            render={<span role="img" aria-label={`${label} info`} className="inline-flex text-muted-foreground" />}
          >
            <Info className="size-3.5" />
          </TooltipTrigger>
          <TooltipContent>
            {tooltip}{" "}
            <a href={docsUrl} target="_blank" rel="noreferrer" className="underline">
              Docs
            </a>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    </span>
  );
}

function MetricsSection({ logEntry, metadata }: { logEntry: LogEntry; metadata: Record<string, any> }) {
  const completionStartTime = logEntry.completionStartTime;
  const ttftMs =
    completionStartTime && completionStartTime !== logEntry.endTime
      ? new Date(completionStartTime).getTime() - new Date(logEntry.startTime).getTime()
      : null;

  const responseCacheValue = String(logEntry.cache_hit ?? "").toLowerCase();
  const isResponseCacheHit = responseCacheValue === "true";
  const showResponseCache = isResponseCacheHit || responseCacheValue === "false";
  const promptCacheReadTokens = Number(metadata?.additional_usage_values?.cache_read_input_tokens) || 0;
  const promptCacheCreationTokens = Number(metadata?.additional_usage_values?.cache_creation_input_tokens) || 0;

  const uncachedInputTokens = getUncachedInputTextTokens(metadata);
  const showAnthropicMessagesInputOutput =
    logEntry.call_type === "anthropic_messages" && uncachedInputTokens !== undefined;

  return (
    <div className="bg-card rounded-lg shadow-sm w-full max-w-full overflow-hidden mb-6">
      <Card size="sm" style={{ marginBottom: 0 }}>
        <CardHeader>
          <CardTitle>Metrics</CardTitle>
        </CardHeader>
        <CardContent>
          <DescriptionList>
            {showAnthropicMessagesInputOutput ? (
              <>
                <DescriptionItem label="Input Tokens">{formatNumberWithCommas(uncachedInputTokens)}</DescriptionItem>
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
            <DescriptionItem label="Duration">
              {logEntry.request_duration_ms != null ? (logEntry.request_duration_ms / 1000).toFixed(3) : "-"} s
            </DescriptionItem>
            {ttftMs != null && ttftMs > 0 && (
              <DescriptionItem label="Time to First Token">{(ttftMs / 1000).toFixed(3)} s</DescriptionItem>
            )}

            {showResponseCache && (
              <DescriptionItem
                label={
                  <MetricLabel
                    label="Response Cache"
                    tooltip={RESPONSE_CACHE_TOOLTIP}
                    docsUrl={RESPONSE_CACHE_DOCS_URL}
                  />
                }
              >
                <Badge variant="secondary" className={isResponseCacheHit ? "bg-success/15 text-success" : undefined}>
                  {isResponseCacheHit ? "Hit" : "Miss"}
                </Badge>
              </DescriptionItem>
            )}
            {promptCacheReadTokens > 0 && (
              <DescriptionItem
                label={
                  <MetricLabel
                    label="Prompt Cache Read Tokens"
                    tooltip={PROMPT_CACHE_READ_TOOLTIP}
                    docsUrl={PROMPT_CACHE_DOCS_URL}
                  />
                }
              >
                {formatNumberWithCommas(promptCacheReadTokens)}
              </DescriptionItem>
            )}
            {promptCacheCreationTokens > 0 && (
              <DescriptionItem
                label={
                  <MetricLabel
                    label="Prompt Cache Creation Tokens"
                    tooltip={PROMPT_CACHE_CREATION_TOOLTIP}
                    docsUrl={PROMPT_CACHE_DOCS_URL}
                  />
                }
              >
                {formatNumberWithCommas(promptCacheCreationTokens)}
              </DescriptionItem>
            )}

            {metadata?.litellm_overhead_time_ms !== undefined && metadata.litellm_overhead_time_ms !== null && (
              <DescriptionItem label="LiteLLM Overhead">
                {metadata.litellm_overhead_time_ms.toFixed(2)} ms
              </DescriptionItem>
            )}

            <DescriptionItem label="Retries">
              {metadata?.attempted_retries !== undefined && metadata?.attempted_retries !== null ? (
                metadata.attempted_retries > 0 ? (
                  <>
                    {metadata.attempted_retries}
                    {metadata.max_retries !== undefined && metadata.max_retries !== null
                      ? ` / ${metadata.max_retries}`
                      : ""}
                  </>
                ) : (
                  <Badge variant="secondary" className="bg-success/15 text-success">
                    None
                  </Badge>
                )
              ) : (
                "-"
              )}
            </DescriptionItem>

            <DescriptionItem label="Start Time">
              {moment(logEntry.startTime).format("YYYY-MM-DDTHH:mm:ss.SSS[Z]")}
            </DescriptionItem>
            <DescriptionItem label="End Time">
              {moment(logEntry.endTime).format("YYYY-MM-DDTHH:mm:ss.SSS[Z]")}
            </DescriptionItem>
          </DescriptionList>
        </CardContent>
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
  const [open, setOpen] = useState(true);
  const [activeTab, setActiveTab] = useState<typeof TAB_REQUEST | typeof TAB_RESPONSE>(TAB_REQUEST);
  const [viewMode, setViewMode] = useState<"pretty" | "json">("pretty");

  const getCopyText = () => {
    const data = activeTab === TAB_REQUEST ? getRawRequest() : getFormattedResponse();
    return JSON.stringify(data, null, 2);
  };

  const totalSpend = logEntry.spend ?? 0;
  const promptTokens = logEntry.prompt_tokens || 0;
  const completionTokens = logEntry.completion_tokens || 0;
  const totalTokens = promptTokens + completionTokens;
  const costBreakdown = logEntry.metadata?.cost_breakdown;
  const useCostBreakdown = costBreakdown?.input_cost !== undefined && costBreakdown?.output_cost !== undefined;
  const inputCost = useCostBreakdown
    ? costBreakdown!.input_cost ?? 0
    : totalTokens > 0
      ? (totalSpend * promptTokens) / totalTokens
      : 0;
  const outputCost = useCostBreakdown
    ? costBreakdown!.output_cost ?? 0
    : totalTokens > 0
      ? (totalSpend * completionTokens) / totalTokens
      : 0;

  return (
    <div className="bg-card rounded-lg shadow-sm w-full max-w-full overflow-hidden mb-6">
      <Collapsible open={open} onOpenChange={setOpen}>
        <Tabs value={viewMode} onValueChange={(value) => setViewMode(value as "pretty" | "json")}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", width: "100%" }}>
            <CollapsibleTrigger className="flex flex-1 items-center gap-3 px-4 py-3 text-left">
              {open ? (
                <ChevronDown className="size-3.5 shrink-0 text-muted-foreground" />
              ) : (
                <ChevronRight className="size-3.5 shrink-0 text-muted-foreground" />
              )}
              <h3 className="text-lg font-medium text-foreground" style={{ margin: 0 }}>
                Request & Response
              </h3>
            </CollapsibleTrigger>
            <TabsList className="mr-4">
              <TabsTrigger value="pretty">Pretty</TabsTrigger>
              <TabsTrigger value="json">JSON</TabsTrigger>
            </TabsList>
          </div>
          <CollapsibleContent>
            <div>
              <TabsContent value="pretty">
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
              </TabsContent>
              <TabsContent value="json">
                <Tabs
                  value={activeTab}
                  onValueChange={(key) => setActiveTab(key as typeof TAB_REQUEST | typeof TAB_RESPONSE)}
                >
                  <div className="flex items-center justify-between">
                    <TabsList>
                      <TabsTrigger value={TAB_REQUEST}>Request</TabsTrigger>
                      <TabsTrigger value={TAB_RESPONSE}>Response</TabsTrigger>
                    </TabsList>
                    <CopyButton
                      getText={getCopyText}
                      label="Copy JSON"
                      disabled={activeTab === TAB_RESPONSE && !hasResponse && !hasError}
                    />
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
                        <div
                          style={{
                            textAlign: "center",
                            padding: 20,
                            color: "var(--color-muted-foreground)",
                            fontStyle: "italic",
                          }}
                        >
                          Response data not available
                        </div>
                      )}
                    </div>
                  </TabsContent>
                </Tabs>
              </TabsContent>
            </div>
          </CollapsibleContent>
        </Tabs>
      </Collapsible>
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
        className={
          allPassed
            ? "border border-success/20 bg-success/10 text-success"
            : "border border-destructive/20 bg-destructive/10 text-destructive"
        }
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: 6,
          padding: "4px 12px",
          borderRadius: 16,
          cursor: "pointer",
          fontSize: 13,
          fontWeight: 500,
        }}
      >
        {allPassed ? "\u2713" : "\u2717"} {guardrailEntries.length} guardrail{guardrailEntries.length !== 1 ? "s" : ""}{" "}
        evaluated
        <span style={{ fontSize: 11, opacity: 0.7 }}>{"\u2193"}</span>
      </div>
    </div>
  );
}

function MetadataSection({ metadata }: { metadata: Record<string, any> }) {
  const [open, setOpen] = useState(true);

  return (
    <div className="bg-card rounded-lg shadow-sm w-full max-w-full overflow-hidden mb-6">
      <Collapsible open={open} onOpenChange={setOpen}>
        <CollapsibleTrigger className="flex w-full items-center gap-3 px-4 py-3 text-left">
          {open ? (
            <ChevronDown className="size-3.5 shrink-0 text-muted-foreground" />
          ) : (
            <ChevronRight className="size-3.5 shrink-0 text-muted-foreground" />
          )}
          <h3 className="text-lg font-medium text-foreground">Metadata</h3>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div>
            <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 8 }}>
              <CopyButton getText={() => JSON.stringify(metadata, null, 2)} label="Copy Metadata" />
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
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}
