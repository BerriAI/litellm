/**
 * RealtimePrettyView - Structured pretty view for OpenAI Realtime API logs
 * Displays session config, conversation turns, and token usage
 * in a readable format instead of raw JSON.
 */

import { useState } from 'react';
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Volume2 as SoundOutlined, MessageSquare as MessageOutlined, Settings as SettingOutlined, Mic as AudioOutlined, ChevronDown as DownOutlined, ChevronUp as UpOutlined } from "lucide-react";
import { SectionHeader } from './SectionHeader';

/** Mimics antd `Typography.Text`'s `type="secondary"` mutted-gray style. */
function SecondaryText({
  children,
  className = "",
  style,
}: {
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
}) {
  return (
    <span className={`text-muted-foreground ${className}`} style={style}>
      {children}
    </span>
  );
}

interface RealtimeEvent {
  type: string;
  event_id?: string;
  session?: RealtimeSession;
  response?: RealtimeResponse;
}

interface RealtimeSession {
  id: string;
  model: string;
  voice?: string;
  modalities?: string[];
  temperature?: number;
  tools?: any[];
  instructions?: string;
  turn_detection?: Record<string, any>;
  input_audio_format?: string;
  output_audio_format?: string;
  max_response_output_tokens?: string | number;
  [key: string]: any;
}

interface RealtimeResponse {
  id: string;
  status: string;
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
    total_tokens?: number;
    input_token_details?: Record<string, any>;
    output_token_details?: Record<string, any>;
  };
  output?: RealtimeOutputItem[];
  modalities?: string[];
  voice?: string;
  conversation_id?: string;
  [key: string]: any;
}

interface RealtimeOutputItem {
  id: string;
  role: string;
  type: string;
  status?: string;
  content?: Array<{
    type: string;
    transcript?: string;
    text?: string;
  }>;
}

interface RealtimePrettyViewProps {
  response: any;
  metrics?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    input_cost?: number;
    output_cost?: number;
  };
}

export function isRealtimeResponse(response: any): boolean {
  if (!response || !response.results || !Array.isArray(response.results) || response.results.length === 0) {
    return false;
  }

  return response.results.some(
    (r: any) =>
      r.type === 'session.created' ||
      r.type === 'session.updated' ||
      r.type === 'response.done'
  );
}

export function RealtimePrettyView({ response, metrics }: RealtimePrettyViewProps) {
  const events: RealtimeEvent[] = response?.results || [];
  const usage = response?.usage;

  const sessionEvent = events.find(
    (e) => e.type === 'session.created' || e.type === 'session.updated'
  );
  const responseEvents = events.filter((e) => e.type === 'response.done');

  return (
    <div>
      {/* Session Configuration Card */}
      {sessionEvent?.session && (
        <SessionCard session={sessionEvent.session} turnCount={responseEvents.length} />
      )}

      {/* Conversation Turns */}
      {responseEvents.length > 0 && (
        <ConversationCard
          responses={responseEvents.map((e) => e.response!).filter(Boolean)}
          totalUsage={usage}
          metrics={metrics}
        />
      )}

      {/* Fallback if no recognized events */}
      {!sessionEvent && responseEvents.length === 0 && (
        <div
          style={{
            border: '1px solid #f0f0f0',
            borderRadius: 6,
            padding: '16px',
            color: '#8c8c8c',
            fontStyle: 'italic',
            fontSize: 13,
          }}
        >
          No recognized realtime events found
        </div>
      )}
    </div>
  );
}

function SessionCard({ session, turnCount }: { session: RealtimeSession; turnCount: number }) {
  const [isCollapsed, setIsCollapsed] = useState(true);

  return (
    <div
      style={{
        border: '1px solid #f0f0f0',
        borderRadius: 6,
        marginBottom: 8,
        overflow: 'hidden',
      }}
    >
      <div
        onClick={() => setIsCollapsed(!isCollapsed)}
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '10px 16px',
          borderBottom: isCollapsed ? 'none' : '1px solid #f0f0f0',
          background: '#fafafa',
          cursor: 'pointer',
          transition: 'background 0.15s ease',
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.background = '#f5f5f5';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.background = '#fafafa';
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center' }}>
            {isCollapsed ? (
              <DownOutlined style={{ fontSize: 10, color: '#8c8c8c' }} />
            ) : (
              <UpOutlined style={{ fontSize: 10, color: '#8c8c8c' }} />
            )}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <SettingOutlined style={{ color: '#8c8c8c', fontSize: 14 }} />
            <span style={{ fontWeight: 500, fontSize: 14 }}>Session</span>
          </div>
          <SecondaryText style={{ fontSize: 12 }}>
            {session.model}
          </SecondaryText>
          {turnCount > 0 && (
            <Badge className="bg-purple-100 text-purple-700 hover:bg-purple-100 font-medium">
              {turnCount} {turnCount === 1 ? 'turn' : 'turns'}
            </Badge>
          )}
          {session.voice && (
            <Badge className="bg-blue-100 text-blue-700 hover:bg-blue-100 gap-1">
              <SoundOutlined size={12} /> {session.voice}
            </Badge>
          )}
          {session.modalities && (
            <div style={{ display: 'flex', gap: 4 }}>
              {session.modalities.map((m) => (
                <Badge key={m} variant="secondary" className="gap-1">
                  {m === 'audio' ? <AudioOutlined size={12} /> : <MessageOutlined size={12} />} {m}
                </Badge>
              ))}
            </div>
          )}
        </div>
      </div>

      <div
        style={{
          maxHeight: isCollapsed ? '0px' : '10000px',
          overflow: 'hidden',
          transition: 'max-height 0.3s ease-out, opacity 0.3s ease-out',
          opacity: isCollapsed ? 0 : 1,
        }}
      >
        <div style={{ padding: '12px 16px' }}>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: '8px 24px',
              fontSize: 13,
            }}
          >
            <ConfigRow label="Model" value={session.model} />
            <ConfigRow label="Voice" value={session.voice} />
            <ConfigRow label="Temperature" value={session.temperature} />
            <ConfigRow
              label="Max Output Tokens"
              value={session.max_response_output_tokens}
            />
            <ConfigRow
              label="Input Audio Format"
              value={session.input_audio_format}
            />
            <ConfigRow
              label="Output Audio Format"
              value={session.output_audio_format}
            />
            {session.turn_detection && (
              <ConfigRow
                label="Turn Detection"
                value={session.turn_detection.type}
              />
            )}
            {session.tools && session.tools.length > 0 && (
              <ConfigRow
                label="Tools"
                value={`${session.tools.length} tool(s)`}
              />
            )}
          </div>

          {session.instructions && (
            <div style={{ marginTop: 12 }}>
              <SecondaryText
                style={{
                  fontSize: 10,
                  letterSpacing: '0.5px',
                  textTransform: 'uppercase',
                  display: 'block',
                  marginBottom: 4,
                }}
              >
                Instructions
              </SecondaryText>
              <div
                style={{
                  fontSize: 12,
                  lineHeight: 1.6,
                  color: '#595959',
                  background: '#fafafa',
                  padding: '8px 12px',
                  borderRadius: 4,
                  border: '1px solid #f0f0f0',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                  maxHeight: 120,
                  overflowY: 'auto',
                }}
              >
                {session.instructions}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function ConversationCard({
  responses,
  totalUsage,
  metrics,
}: {
  responses: RealtimeResponse[];
  totalUsage?: any;
  metrics?: RealtimePrettyViewProps['metrics'];
}) {
  const [isCollapsed, setIsCollapsed] = useState(false);

  const totalTokens = totalUsage?.total_tokens;
  const turnCount = responses.length;
  const handleCopy = () => {
    const transcripts = responses
      .flatMap((r) =>
        (r.output || []).flatMap((o) =>
          (o.content || []).map(
            (c) => `${o.role}: ${c.transcript || c.text || ''}`
          )
        )
      )
      .join('\n');
    navigator.clipboard.writeText(transcripts);
  };

  return (
    <div
      style={{
        border: '1px solid #f0f0f0',
        borderRadius: 6,
        overflow: 'hidden',
      }}
    >
      <SectionHeader
        type="output"
        tokens={metrics?.completion_tokens ?? totalTokens}
        cost={metrics?.output_cost}
        onCopy={handleCopy}
        isCollapsed={isCollapsed}
        onToggleCollapse={() => setIsCollapsed(!isCollapsed)}
        turnCount={turnCount}
      />

      <div
        style={{
          maxHeight: isCollapsed ? '0px' : '10000px',
          overflow: 'hidden',
          transition: 'max-height 0.3s ease-out, opacity 0.3s ease-out',
          opacity: isCollapsed ? 0 : 1,
        }}
      >
        <div style={{ padding: '12px 16px' }}>
          {responses.map((resp, idx) => (
            <ResponseTurn key={resp.id || idx} response={resp} index={idx} />
          ))}
        </div>
      </div>
    </div>
  );
}

function ResponseTurn({
  response,
  index,
}: {
  response: RealtimeResponse;
  index: number;
}) {
  const outputs = response.output || [];
  const usage = response.usage;

  return (
    <div
      style={{
        marginBottom: 12,
        paddingBottom: 12,
        borderBottom: '1px solid #f5f5f5',
      }}
    >
      {/* Turn header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          marginBottom: 8,
        }}
      >
        <Badge
          className={
            response.status === 'completed'
              ? "bg-green-100 text-green-700 hover:bg-green-100"
              : "bg-orange-100 text-orange-700 hover:bg-orange-100"
          }
        >
          {response.status || 'unknown'}
        </Badge>
        {usage && (
          <SecondaryText style={{ fontSize: 11 }}>
            {usage.input_tokens ?? 0} in / {usage.output_tokens ?? 0} out tokens
          </SecondaryText>
        )}
        {response.conversation_id && (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="text-muted-foreground cursor-help" style={{ fontSize: 11 }}>
                  conv: {response.conversation_id.slice(0, 12)}...
                </span>
              </TooltipTrigger>
              <TooltipContent>{response.conversation_id}</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}
      </div>

      {/* Output messages / transcripts */}
      {outputs.map((output, oIdx) => (
        <OutputMessage key={output.id || oIdx} output={output} />
      ))}

      {/* Token breakdown if available */}
      {usage?.input_token_details && (
        <TokenBreakdown
          label="Input"
          details={usage.input_token_details}
        />
      )}
      {usage?.output_token_details && (
        <TokenBreakdown
          label="Output"
          details={usage.output_token_details}
        />
      )}
    </div>
  );
}

function OutputMessage({ output }: { output: RealtimeOutputItem }) {
  const contents = output.content || [];
  const hasTranscripts = contents.some((c) => c.transcript || c.text);

  if (!hasTranscripts) return null;

  return (
    <div style={{ marginBottom: 8 }}>
      <SecondaryText
        style={{
          fontSize: 10,
          letterSpacing: '0.5px',
          textTransform: 'uppercase',
          display: 'block',
          marginBottom: 3,
        }}
      >
        {output.role?.toUpperCase() || 'ASSISTANT'}
      </SecondaryText>
      {contents.map((c, cIdx) => {
        const text = c.transcript || c.text;
        if (!text) return null;
        return (
          <div
            key={cIdx}
            style={{
              display: 'flex',
              alignItems: 'flex-start',
              gap: 8,
              marginBottom: 4,
            }}
          >
            {c.type === 'audio' && (
              <AudioOutlined
                style={{
                  color: '#8c8c8c',
                  fontSize: 12,
                  marginTop: 3,
                  flexShrink: 0,
                }}
              />
            )}
            {c.type === 'text' && (
              <MessageOutlined
                style={{
                  color: '#8c8c8c',
                  fontSize: 12,
                  marginTop: 3,
                  flexShrink: 0,
                }}
              />
            )}
            <div
              style={{
                fontSize: 13,
                lineHeight: 1.7,
                color: '#262626',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
              }}
            >
              {text}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function TokenBreakdown({
  label,
  details,
}: {
  label: string;
  details: Record<string, any>;
}) {
  const entries = Object.entries(details).filter(
    ([, v]) =>
      typeof v === 'number' ||
      (typeof v === 'object' && v !== null)
  );

  if (entries.length === 0) return null;

  return (
    <div style={{ marginTop: 4 }}>
      <SecondaryText
        style={{ fontSize: 10, letterSpacing: '0.5px', textTransform: 'uppercase' }}
      >
        {label} Token Breakdown
      </SecondaryText>
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: 8,
          marginTop: 4,
        }}
      >
        {entries.map(([key, value]) => {
          if (typeof value === 'number') {
            return (
              <Badge key={key} variant="secondary">
                {formatTokenLabel(key)}: {value.toLocaleString()}
              </Badge>
            );
          }
          return null;
        })}
      </div>
    </div>
  );
}

function ConfigRow({
  label,
  value,
}: {
  label: string;
  value: any;
}) {
  if (value === undefined || value === null) return null;
  return (
    <div>
      <SecondaryText style={{ fontSize: 11 }}>
        {label}
      </SecondaryText>
      <div style={{ fontSize: 13 }} className="text-foreground">
        {String(value)}
      </div>
    </div>
  );
}

function formatTokenLabel(key: string): string {
  return key
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}
