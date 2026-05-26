"use client";

import React, { useState } from "react";
import {
  Alert,
  Button,
  Checkbox,
  Collapse,
  Empty,
  Input,
  Space,
  Spin,
  Switch,
  Tag,
  Tooltip,
  Typography,
} from "antd";
// Empty is used in the skills panel below.
import {
  CheckCircleTwoTone,
  InfoCircleOutlined,
  LinkOutlined,
  ReloadOutlined,
  SearchOutlined,
} from "@ant-design/icons";

import {
  DiscoveredAgentCard,
  DiscoveryMode,
  discoverAgentCardCall,
} from "../networking";

const { Text, Paragraph } = Typography;
const { Panel } = Collapse;

export interface DiscoveredAgentCardSelection {
  /** Full upstream card the proxy fetched, unmodified. */
  raw_card: DiscoveredAgentCard;
  /** Subset of the upstream card with only the user-selected skills and
   *  capabilities, plus the user-edited name/description. Suitable to send as
   *  ``agent_card_params`` on ``POST /v1/agents``. */
  selected_card: DiscoveredAgentCard;
  /** The base URL the user pasted in. */
  upstream_url: string;
}

/**
 * What the parent wants the discovery endpoint to do. When the parent can
 * derive this from form state (e.g. agent type = LangGraph + assistant_id +
 * api_base), it owns the values; the component just relays them. Different
 * upstreams use different URL conventions, so the mode matters.
 */
export interface DiscoveryRequestPlan {
  /** Base URL to send to the proxy. */
  url: string;
  /** Which dispatch path the proxy should use. */
  discovery_mode: DiscoveryMode;
  /** Mode-specific params (e.g. ``{assistant_id}`` for LangGraph). */
  params?: Record<string, any>;
  /** Human-readable rendering of the URL the proxy will ultimately fetch.
   *  Shown in the UI so the admin can see what we'll hit. */
  display_url?: string;
}

interface AgentCardDiscoveryProps {
  accessToken: string | null;
  onApply: (selection: DiscoveredAgentCardSelection) => void;
  /**
   * Parent-supplied discovery plan. When provided the component uses these
   * values verbatim and hides its free-form URL input — the parent is the
   * source of truth (e.g. for LangGraph it's derived from api_base +
   * assistant_id). When omitted the component falls back to a manual URL
   * input that defaults to ``well_known_fallback`` mode.
   */
  discoveryRequest?: DiscoveryRequestPlan;
}

const ALLOWED_CAPABILITY_KEYS = ["streaming"] as const;

const skillId = (skill: any, idx: number): string =>
  skill?.id ?? skill?.name ?? `skill-${idx}`;

/**
 * Mirrors the proxy-side `_ALLOWED_CAPABILITY_KEYS` allowlist. Keep these in
 * sync — anything we surface here that the proxy strips would look like a
 * silent drop to the admin.
 */
const filterCapabilitiesForUI = (
  capabilities: Record<string, any> | undefined,
): Record<string, any> => {
  if (!capabilities) return {};
  return ALLOWED_CAPABILITY_KEYS.reduce<Record<string, any>>((acc, key) => {
    if (key in capabilities) acc[key] = Boolean(capabilities[key]);
    return acc;
  }, {});
};

const AgentCardDiscovery: React.FC<AgentCardDiscoveryProps> = ({
  accessToken,
  onApply,
  discoveryRequest,
}) => {
  // When the parent drives discovery, ``manualUrl`` is unused — the URL
  // comes from ``discoveryRequest.url`` directly. When the parent hasn't
  // supplied a plan, the admin types into this field manually.
  const [manualUrl, setManualUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [card, setCard] = useState<DiscoveredAgentCard | null>(null);

  const isParentDriven = discoveryRequest !== undefined;
  const effectiveUrl = isParentDriven ? discoveryRequest!.url : manualUrl;

  const [editedName, setEditedName] = useState<string>("");
  const [editedDescription, setEditedDescription] = useState<string>("");
  const [selectedSkillIds, setSelectedSkillIds] = useState<Set<string>>(new Set());
  const [selectedCapabilities, setSelectedCapabilities] = useState<
    Record<string, boolean>
  >({});

  const resetSelections = (fresh: DiscoveredAgentCard) => {
    setEditedName(fresh.name ?? "");
    setEditedDescription(fresh.description ?? "");
    const skills = fresh.skills ?? [];
    setSelectedSkillIds(new Set(skills.map((s, i) => skillId(s, i))));
    setSelectedCapabilities(filterCapabilitiesForUI(fresh.capabilities));
  };

  const handleDiscover = async () => {
    if (!accessToken) {
      setError("No access token available");
      return;
    }
    const trimmed = effectiveUrl.trim();
    if (!trimmed) {
      setError(
        isParentDriven
          ? "Fill in the agent's connection details above first"
          : "Enter the agent's base URL first",
      );
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await discoverAgentCardCall(
        accessToken,
        trimmed,
        isParentDriven
          ? {
              discovery_mode: discoveryRequest!.discovery_mode,
              params: discoveryRequest!.params,
            }
          : undefined,
      );
      setCard(response.agent_card);
      resetSelections(response.agent_card);
    } catch (e: any) {
      setError(e?.message ? String(e.message) : "Failed to discover agent card");
      setCard(null);
    } finally {
      setLoading(false);
    }
  };

  const toggleSkill = (id: string, checked: boolean) => {
    setSelectedSkillIds((prev) => {
      const next = new Set(prev);
      if (checked) next.add(id);
      else next.delete(id);
      return next;
    });
  };

  const handleApply = () => {
    if (!card) return;
    const skills = card.skills ?? [];
    const filteredSkills = skills.filter((s, i) =>
      selectedSkillIds.has(skillId(s, i)),
    );

    const selected_card: DiscoveredAgentCard = {
      ...card,
      name: editedName,
      description: editedDescription,
      skills: filteredSkills,
      capabilities: { ...selectedCapabilities },
    };

    onApply({
      raw_card: card,
      selected_card,
      upstream_url: effectiveUrl.trim(),
    });
  };

  const skillCount = card?.skills?.length ?? 0;
  const selectedSkillCount = selectedSkillIds.size;

  return (
    <div className="border border-gray-200 rounded-lg p-4 bg-gray-50 mb-4">
      <div className="flex items-center gap-2 mb-2">
        <LinkOutlined className="text-indigo-600" />
        <Text strong>Discover from agent URL</Text>
        <Tooltip title="LiteLLM will fetch /.well-known/agent-card.json from this URL and let you pick which skills and capabilities to expose through the proxy.">
          <InfoCircleOutlined className="text-gray-400" />
        </Tooltip>
      </div>
      {isParentDriven ? (
        <>
          <Paragraph className="text-xs text-gray-500 mb-2">
            Using the connection details you entered above. We&apos;ll fetch:
          </Paragraph>
          <div className="bg-white border border-gray-200 rounded px-3 py-2 mb-3 font-mono text-xs text-gray-700 break-all">
            {discoveryRequest!.display_url || effectiveUrl || (
              <span className="text-gray-400 italic">
                Fill in the fields above first
              </span>
            )}
          </div>
          <div className="flex justify-end">
            <Button
              type="primary"
              icon={card ? <ReloadOutlined /> : <SearchOutlined />}
              loading={loading}
              onClick={handleDiscover}
              disabled={!effectiveUrl.trim()}
            >
              {card ? "Re-discover" : "Discover"}
            </Button>
          </div>
        </>
      ) : (
        <>
          <Paragraph className="text-xs text-gray-500 mb-3">
            Paste the upstream agent&apos;s base URL. We&apos;ll try{" "}
            <code>/.well-known/agent-card.json</code>,{" "}
            <code>/.well-known/agent.json</code>, and <code>/agent.json</code>{" "}
            in order.
          </Paragraph>

          <Space.Compact style={{ width: "100%" }}>
            <Input
              placeholder="https://upstream-agent.example.com"
              value={manualUrl}
              onChange={(e) => setManualUrl(e.target.value)}
              onPressEnter={handleDiscover}
              allowClear
              disabled={loading}
            />
            <Button
              type="primary"
              icon={card ? <ReloadOutlined /> : <SearchOutlined />}
              loading={loading}
              onClick={handleDiscover}
            >
              {card ? "Re-discover" : "Discover"}
            </Button>
          </Space.Compact>
        </>
      )}

      {error && (
        <Alert
          className="mt-3"
          type="error"
          message="Discovery failed"
          description={error}
          showIcon
          closable
          onClose={() => setError(null)}
        />
      )}

      {loading && !card && (
        <div className="flex items-center justify-center py-8">
          <Spin />
        </div>
      )}

      {card && (
        <div className="mt-4 bg-white border border-gray-200 rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <Space>
              <CheckCircleTwoTone twoToneColor="#52c41a" />
              <Text strong>Upstream card loaded</Text>
              {card.version && <Tag color="blue">v{card.version}</Tag>}
              {card.provider?.organization && (
                <Tag color="purple">{card.provider.organization}</Tag>
              )}
            </Space>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
            <div>
              <label className="text-xs font-medium text-gray-600 block mb-1">
                Name (shown to API clients)
              </label>
              <Input
                value={editedName}
                onChange={(e) => setEditedName(e.target.value)}
                placeholder="Agent name"
              />
            </div>
            <div>
              <label className="text-xs font-medium text-gray-600 block mb-1">
                Description
              </label>
              <Input.TextArea
                value={editedDescription}
                onChange={(e) => setEditedDescription(e.target.value)}
                rows={2}
                placeholder="What this agent does"
              />
            </div>
          </div>

          <Collapse
            defaultActiveKey={["skills", "capabilities"]}
            ghost
            className="bg-transparent"
          >
            <Panel
              key="skills"
              header={
                <Space>
                  <Text strong>Skills</Text>
                  <Tag>
                    {selectedSkillCount} / {skillCount} selected
                  </Tag>
                </Space>
              }
            >
              {skillCount === 0 ? (
                <Empty
                  image={Empty.PRESENTED_IMAGE_SIMPLE}
                  description="Upstream card has no skills"
                />
              ) : (
                <div className="space-y-2">
                  {(card.skills ?? []).map((skill, idx) => {
                    const id = skillId(skill, idx);
                    const checked = selectedSkillIds.has(id);
                    return (
                      <label
                        key={id}
                        className={`flex items-start gap-3 p-3 border rounded cursor-pointer transition-colors ${
                          checked
                            ? "border-indigo-300 bg-indigo-50"
                            : "border-gray-200 bg-white hover:border-gray-300"
                        }`}
                      >
                        <Checkbox
                          checked={checked}
                          onChange={(e) => toggleSkill(id, e.target.checked)}
                        />
                        <div className="flex-1">
                          <div className="flex items-center gap-2 flex-wrap">
                            <Text strong>{skill.name || id}</Text>
                            {skill.id && (
                              <Tag style={{ marginLeft: 0 }}>{skill.id}</Tag>
                            )}
                            {(skill.tags ?? []).map((t: string) => (
                              <Tag key={t} color="geekblue">
                                {t}
                              </Tag>
                            ))}
                          </div>
                          {skill.description && (
                            <Paragraph
                              className="text-xs text-gray-500 mt-1 mb-0"
                              ellipsis={{ rows: 2, expandable: true, symbol: "more" }}
                            >
                              {skill.description}
                            </Paragraph>
                          )}
                        </div>
                      </label>
                    );
                  })}
                </div>
              )}
            </Panel>

            <Panel
              key="capabilities"
              header={
                <Space>
                  <Text strong>Capabilities</Text>
                  <Tooltip title="Only capabilities LiteLLM can faithfully proxy today are listed. Others (push notifications, extensions) are coming soon.">
                    <InfoCircleOutlined className="text-gray-400" />
                  </Tooltip>
                </Space>
              }
            >
              <div className="space-y-2">
                {ALLOWED_CAPABILITY_KEYS.map((key) => {
                  const upstreamHas = Boolean(card.capabilities?.[key]);
                  return (
                    <div
                      key={key}
                      className="flex items-center justify-between p-2 border border-gray-200 rounded bg-white"
                    >
                      <div>
                        <Text strong className="capitalize">
                          {key}
                        </Text>
                        {!upstreamHas && (
                          <Tag className="ml-2" color="default">
                            not advertised upstream
                          </Tag>
                        )}
                      </div>
                      <Switch
                        checked={Boolean(selectedCapabilities[key])}
                        onChange={(checked) =>
                          setSelectedCapabilities((prev) => ({
                            ...prev,
                            [key]: checked,
                          }))
                        }
                      />
                    </div>
                  );
                })}
              </div>
            </Panel>
          </Collapse>

          <div className="flex justify-end mt-4">
            <Button type="primary" onClick={handleApply}>
              Use these selections
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};

export default AgentCardDiscovery;
