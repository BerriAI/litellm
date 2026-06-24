/**
 * Labeled token metric heading with an info tooltip.
 *
 * Centralizes the wording for what "input", "output", "cache read", and
 * "cache write" tokens actually mean so the Cost tab (and, later, the other
 * usage views) communicate token accounting consistently.
 *
 * Backend semantics these definitions reflect:
 * - `prompt_tokens` is the *total* input, already including cache read +
 *   cache creation tokens (LiteLLM normalizes Anthropic's split usage to
 *   match OpenAI's convention — see anthropic/chat/transformation.py).
 * - `completion_tokens` is all output, including reasoning/thinking tokens.
 * - `total_tokens` = prompt_tokens + completion_tokens, so cache tokens are a
 *   subset of input, never added on top.
 * - The "Input Tokens" card shows the non-cached portion
 *   (prompt_tokens − cache_read − cache_write), so Input + Output + Cache
 *   Read + Cache Write sum exactly to Total Tokens.
 */

import { InfoCircleOutlined } from "@ant-design/icons";
import { Title } from "@tremor/react";
import { Tooltip } from "antd";
import React from "react";

export type TokenMetricKind = "total" | "input" | "output" | "cache_read" | "cache_write";

const TOOLTIP_COPY: Record<TokenMetricKind, React.ReactNode> = {
  total: (
    <>
      Total = Input + Output. Cached tokens are counted as part of Input (not added on top), so the Input, Output, Cache
      Read, and Cache Write cards sum exactly to this number. Reflects successful requests only.
    </>
  ),
  input: (
    <>
      Non-cached prompt tokens sent to the model — the prompt minus any tokens served from or written to the prompt
      cache. Billed at the model&apos;s standard input rate. Cached portions are shown separately as Cache Read / Cache
      Write.
    </>
  ),
  output: (
    <>
      Tokens the model generated in its response, including any reasoning / &ldquo;thinking&rdquo; tokens. Billed at the
      model&apos;s output rate.
    </>
  ),
  cache_read: (
    <>
      Input tokens served from the prompt cache instead of being reprocessed. Billed at a steep discount versus standard
      input (e.g. ~0.1× on Anthropic / OpenAI). Already included in total input.
    </>
  ),
  cache_write: (
    <>
      Input tokens written into the prompt cache so later requests can reuse them. Billed at a premium over standard
      input on some providers (e.g. ~1.25× on Anthropic). Already included in total input.
    </>
  ),
};

interface TokenMetricLabelProps {
  kind: TokenMetricKind;
  label: string;
  /** Optional trailing node (e.g. an expand/collapse chevron) rendered after the icon. */
  trailing?: React.ReactNode;
}

const TokenMetricLabel: React.FC<TokenMetricLabelProps> = ({ kind, label, trailing }) => (
  <div className="flex items-center gap-2">
    <Title>{label}</Title>
    <Tooltip title={TOOLTIP_COPY[kind]}>
      <InfoCircleOutlined className="text-gray-400 hover:text-gray-600" />
    </Tooltip>
    {trailing}
  </div>
);

export default TokenMetricLabel;
