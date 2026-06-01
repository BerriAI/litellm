"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
  discoverAgentCardCall,
} from "../networking";
import {
  ALLOWED_CAPABILITY_KEYS,
  selectionsFromSavedAgentCard,
  selectionsFromUpstreamCard,
  skillId,
} from "./agent_discovery_utils";

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

export type { DiscoveryRequestPlan } from "./agent_discovery_utils";
import type { DiscoveryRequestPlan } from "./agent_discovery_utils";

interface AgentCardDiscoveryProps {
  accessToken: string | null;
  /** Called whenever the upstream card or the user's selections change. Pass
   *  ``null`` when discovery is cleared or fails so the parent can reset. */
  onApply: (selection: DiscoveredAgentCardSelection | null) => void;
  /**
   * Parent-supplied discovery plan. When provided the component uses these
   * values verbatim and hides its free-form URL input — the parent is the
   * source of truth (e.g. for LangGraph it's derived from api_base +
   * assistant_id). When omitted the component falls back to a manual URL
   * input that defaults to ``well_known_fallback`` mode.
   */
  discoveryRequest?: DiscoveryRequestPlan;
  /** When editing an existing agent, the card stored in the DB. Upstream
   *  discovery lists everything available; only skills/capabilities present
   *  here are pre-selected. */
  savedAgentCard?: DiscoveredAgentCard | null;
}

const AgentCardDiscovery: React.FC<AgentCardDiscoveryProps> = ({
  accessToken,
  onApply,
  discoveryRequest,
  savedAgentCard,
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

  const onApplyRef = useRef(onApply);
  onApplyRef.current = onApply;
  const discoverRequestIdRef = useRef(0);
  const lastSyncedSelectionRef = useRef<string | null>(null);
  // Hold the latest ``discoveryRequest`` in a ref so ``handleDiscover`` can
  // read its ``discovery_mode``/``params`` without depending on the object
  // identity itself — the parent recreates the object on every form keystroke
  // even when the underlying values are unchanged. We use stable primitive
  // keys (``discoveryMode`` + ``discoveryParamsKey``) as the actual deps so
  // the callback / effect only re-run when content actually changes.
  const discoveryRequestRef = useRef(discoveryRequest);
  discoveryRequestRef.current = discoveryRequest;
  // Hold ``savedAgentCard`` in a ref so ``resetSelections`` always sees the
  // latest value without making it a dependency of ``handleDiscover``.
  // Putting ``savedAgentCard`` directly in the callback deps means any parent
  // re-render that hands us a new object reference (e.g. a background
  // agent-data refresh during editing) recreates ``handleDiscover``, which
  // re-fires the auto-discover effect and overwrites in-progress user edits.
  const savedAgentCardRef = useRef(savedAgentCard);
  savedAgentCardRef.current = savedAgentCard;

  const resetSelections = (fresh: DiscoveredAgentCard) => {
    const saved = savedAgentCardRef.current;
    const initial = saved
      ? selectionsFromSavedAgentCard(fresh, saved)
      : selectionsFromUpstreamCard(fresh);
    setEditedName(initial.editedName);
    setEditedDescription(initial.editedDescription);
    setSelectedSkillIds(initial.selectedSkillIds);
    setSelectedCapabilities(initial.selectedCapabilities);
  };

  const discoveryMode = discoveryRequest?.discovery_mode;
  const discoveryParamsKey = useMemo(
    () => JSON.stringify(discoveryRequest?.params ?? null),
    [discoveryRequest?.params],
  );

  const handleDiscover = useCallback(async () => {
    if (!accessToken) {
      setError("No access token available");
      onApplyRef.current(null);
      return;
    }
    const trimmed = effectiveUrl.trim();
    if (!trimmed) {
      setError(
        isParentDriven
          ? "Fill in the agent's connection details above first"
          : "Enter the agent's base URL first",
      );
      setCard(null);
      onApplyRef.current(null);
      return;
    }

    const currentDiscoveryRequest = discoveryRequestRef.current;
    const requestId = ++discoverRequestIdRef.current;
    setLoading(true);
    setError(null);
    try {
      const response = await discoverAgentCardCall(
        accessToken,
        trimmed,
        isParentDriven && currentDiscoveryRequest
          ? {
              discovery_mode: currentDiscoveryRequest.discovery_mode,
              params: currentDiscoveryRequest.params,
            }
          : undefined,
      );
      if (requestId !== discoverRequestIdRef.current) return;
      lastSyncedSelectionRef.current = null;
      setCard(response.agent_card);
      resetSelections(response.agent_card);
    } catch (e: any) {
      if (requestId !== discoverRequestIdRef.current) return;
      setError(e?.message ? String(e.message) : "Failed to discover agent card");
      setCard(null);
      lastSyncedSelectionRef.current = null;
      onApplyRef.current(null);
    } finally {
      if (requestId === discoverRequestIdRef.current) {
        setLoading(false);
      }
    }
    // ``discoveryMode`` / ``discoveryParamsKey`` are primitive proxies for
    // ``discoveryRequest`` content; the actual object is read via the ref
    // above so identity churn from the parent doesn't recreate this callback.
    // ``savedAgentCard`` is intentionally NOT a dep — it's read via
    // ``savedAgentCardRef`` inside ``resetSelections``. Including it here
    // would recreate this callback whenever the parent hands us a new
    // ``savedAgentCard`` object (e.g. a background refresh of agent data
    // during editing), which would re-fire the auto-discover effect and
    // wipe in-progress user selections.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    accessToken,
    effectiveUrl,
    isParentDriven,
    discoveryMode,
    discoveryParamsKey,
  ]);

  // Auto-discover when the URL (or parent plan) becomes available. Debounce
  // is applied uniformly so rapid changes from a watched parent form (e.g.
  // typing into a LangGraph api_base / assistant_id field) don't fire one
  // HTTP request per keystroke.
  useEffect(() => {
    if (!accessToken) return;
    const trimmed = effectiveUrl.trim();
    if (!trimmed) {
      setCard(null);
      setError(null);
      lastSyncedSelectionRef.current = null;
      onApplyRef.current(null);
      return;
    }

    const timer = window.setTimeout(() => {
      void handleDiscover();
    }, 400);
    return () => window.clearTimeout(timer);
  }, [accessToken, effectiveUrl, handleDiscover]);

  const toggleSkill = (id: string, checked: boolean) => {
    setSelectedSkillIds((prev) => {
      const next = new Set(prev);
      if (checked) next.add(id);
      else next.delete(id);
      return next;
    });
  };

  const buildSelection = useCallback((): DiscoveredAgentCardSelection | null => {
    if (!card) return null;
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

    return {
      raw_card: card,
      selected_card,
      upstream_url: effectiveUrl.trim(),
    };
  }, [
    card,
    editedDescription,
    editedName,
    effectiveUrl,
    selectedCapabilities,
    selectedSkillIds,
  ]);

  // Keep the parent form in sync as the user edits selections — no extra
  // "apply" click needed before hitting Next.
  useEffect(() => {
    if (!card) return;
    const selection = buildSelection();
    const serialized = JSON.stringify(selection);
    if (lastSyncedSelectionRef.current === serialized) return;
    lastSyncedSelectionRef.current = serialized;
    onApplyRef.current(selection);
  }, [buildSelection, card]);

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

        </div>
      )}
    </div>
  );
};

export default AgentCardDiscovery;
