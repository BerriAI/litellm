"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, CircleAlert, CircleCheck, Info, Link as LinkIcon, RotateCw, Search, X } from "lucide-react";

import { useDebouncedCallback } from "@tanstack/react-pacer/debouncer";
import { DiscoveredAgentCard, discoverAgentCardCall } from "@/components/networking";
import { Alert, AlertAction, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import {
  ALLOWED_CAPABILITY_KEYS,
  selectionsFromSavedAgentCard,
  selectionsFromUpstreamCard,
  skillId,
} from "./agent_discovery_utils";

const DISCOVERY_DEBOUNCE_WAIT_MS = 400;

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
  const [selectedCapabilities, setSelectedCapabilities] = useState<Record<string, boolean>>({});

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
    const initial = saved ? selectionsFromSavedAgentCard(fresh, saved) : selectionsFromUpstreamCard(fresh);
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
        isParentDriven ? "Fill in the agent's connection details above first" : "Enter the agent's base URL first",
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
  }, [accessToken, effectiveUrl, isParentDriven, discoveryMode, discoveryParamsKey]);

  const debouncedDiscover = useDebouncedCallback(
    () => {
      if (!accessToken || !effectiveUrl.trim()) return;
      void handleDiscover();
    },
    { wait: DISCOVERY_DEBOUNCE_WAIT_MS },
  );

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

    debouncedDiscover();
  }, [accessToken, effectiveUrl, handleDiscover, debouncedDiscover]);

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
    const filteredSkills = skills.filter((s, i) => selectedSkillIds.has(skillId(s, i)));

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
  }, [card, editedDescription, editedName, effectiveUrl, selectedCapabilities, selectedSkillIds]);

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

  const renderDiscoverIcon = () => {
    if (loading) return <UiLoadingSpinner className="size-4" />;
    if (card) return <RotateCw />;
    return <Search />;
  };
  const discoverLabel = card ? "Re-discover" : "Discover";

  return (
    <div className="mb-4 rounded-lg border border-border bg-muted/50 p-4">
      <div className="mb-2 flex items-center gap-2">
        <LinkIcon className="size-4 text-primary" />
        <span className="text-sm font-medium text-foreground">Discover from agent URL</span>
        <TooltipProvider delay={300}>
          <Tooltip>
            <TooltipTrigger
              render={
                <span className="inline-flex text-muted-foreground">
                  <Info className="size-4" />
                </span>
              }
            />
            <TooltipContent>
              LiteLLM will fetch /.well-known/agent-card.json from this URL and let you pick which skills and
              capabilities to expose through the proxy.
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
      {isParentDriven ? (
        <>
          <p className="mb-2 text-xs text-muted-foreground">
            Using the connection details you entered above. We&apos;ll fetch:
          </p>
          <div className="mb-3 rounded-sm border border-border bg-background px-3 py-2 font-mono text-xs break-all text-foreground">
            {discoveryRequest!.display_url || effectiveUrl || (
              <span className="text-muted-foreground italic">Fill in the fields above first</span>
            )}
          </div>
          <div className="flex justify-end">
            <Button onClick={handleDiscover} disabled={loading || !effectiveUrl.trim()}>
              {renderDiscoverIcon()}
              {discoverLabel}
            </Button>
          </div>
        </>
      ) : (
        <>
          <p className="mb-3 text-xs text-muted-foreground">
            Paste the upstream agent&apos;s base URL. We&apos;ll try <code>/.well-known/agent-card.json</code>,{" "}
            <code>/.well-known/agent.json</code>, and <code>/agent.json</code> in order.
          </p>

          <div className="flex w-full items-center gap-2">
            <Input
              placeholder="https://upstream-agent.example.com"
              value={manualUrl}
              onChange={(e) => setManualUrl(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleDiscover();
              }}
              disabled={loading}
            />
            <Button onClick={handleDiscover} disabled={loading}>
              {renderDiscoverIcon()}
              {discoverLabel}
            </Button>
          </div>
        </>
      )}

      {error && (
        <Alert variant="destructive" className="mt-3">
          <CircleAlert />
          <AlertTitle>Discovery failed</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
          <AlertAction>
            <Button variant="ghost" size="icon-xs" aria-label="Dismiss error" onClick={() => setError(null)}>
              <X />
            </Button>
          </AlertAction>
        </Alert>
      )}

      {loading && !card && (
        <div className="flex items-center justify-center py-8">
          <UiLoadingSpinner className="size-6 text-muted-foreground" />
        </div>
      )}

      {card && (
        <div className="mt-4 rounded-lg border border-border bg-background p-4">
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <CircleCheck className="size-4 text-green-600" />
            <span className="text-sm font-medium text-foreground">Upstream card loaded</span>
            {card.version && <Badge variant="secondary">v{card.version}</Badge>}
            {card.provider?.organization && <Badge variant="secondary">{card.provider.organization}</Badge>}
          </div>

          <div className="mb-4 grid grid-cols-1 gap-3 md:grid-cols-2">
            <div>
              <label className="mb-1 block text-xs font-medium text-muted-foreground">
                Name (shown to API clients)
              </label>
              <Input value={editedName} onChange={(e) => setEditedName(e.target.value)} placeholder="Agent name" />
            </div>
            <div>
              <label className="mb-1 block text-xs font-medium text-muted-foreground">Description</label>
              <Textarea
                className="field-sizing-fixed min-h-0"
                value={editedDescription}
                onChange={(e) => setEditedDescription(e.target.value)}
                rows={2}
                placeholder="What this agent does"
              />
            </div>
          </div>

          <div className="flex flex-col gap-4">
            <Collapsible defaultOpen>
              <div className="flex items-center gap-2">
                <CollapsibleTrigger
                  render={
                    <button type="button" className="group flex items-center gap-2">
                      <ChevronDown className="size-4 text-muted-foreground transition-transform group-data-[panel-open]:rotate-180" />
                      <span className="text-sm font-medium text-foreground">Skills</span>
                    </button>
                  }
                />
                <Badge variant="secondary">
                  {selectedSkillCount} / {skillCount} selected
                </Badge>
              </div>
              <CollapsibleContent className="pt-2">
                {skillCount === 0 ? (
                  <div className="py-6 text-center text-sm text-muted-foreground">Upstream card has no skills</div>
                ) : (
                  <div className="space-y-2">
                    {(card.skills ?? []).map((skill, idx) => {
                      const id = skillId(skill, idx);
                      const checked = selectedSkillIds.has(id);
                      return (
                        <label
                          key={id}
                          className={`flex cursor-pointer items-start gap-3 rounded border p-3 transition-colors ${
                            checked ? "border-primary/40 bg-primary/5" : "border-border bg-background hover:border-ring"
                          }`}
                        >
                          <Checkbox checked={checked} onCheckedChange={(next) => toggleSkill(id, next)} />
                          <div className="flex-1">
                            <div className="flex flex-wrap items-center gap-2">
                              <span className="text-sm font-medium text-foreground">{skill.name || id}</span>
                              {skill.id && <Badge variant="secondary">{skill.id}</Badge>}
                              {(skill.tags ?? []).map((t: string) => (
                                <Badge key={t} variant="outline">
                                  {t}
                                </Badge>
                              ))}
                            </div>
                            {skill.description && (
                              <p className="mt-1 line-clamp-2 text-xs text-muted-foreground">{skill.description}</p>
                            )}
                          </div>
                        </label>
                      );
                    })}
                  </div>
                )}
              </CollapsibleContent>
            </Collapsible>

            <Collapsible defaultOpen>
              <div className="flex items-center gap-2">
                <CollapsibleTrigger
                  render={
                    <button type="button" className="group flex items-center gap-2">
                      <ChevronDown className="size-4 text-muted-foreground transition-transform group-data-[panel-open]:rotate-180" />
                      <span className="text-sm font-medium text-foreground">Capabilities</span>
                    </button>
                  }
                />
                <TooltipProvider delay={300}>
                  <Tooltip>
                    <TooltipTrigger
                      render={
                        <span className="inline-flex text-muted-foreground">
                          <Info className="size-4" />
                        </span>
                      }
                    />
                    <TooltipContent>
                      Only capabilities LiteLLM can faithfully proxy today are listed. Others (push notifications,
                      extensions) are coming soon.
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
              <CollapsibleContent className="pt-2">
                <div className="space-y-2">
                  {ALLOWED_CAPABILITY_KEYS.map((key) => {
                    const upstreamHas = Boolean(card.capabilities?.[key]);
                    return (
                      <div
                        key={key}
                        className="flex items-center justify-between rounded-sm border border-border bg-background p-2"
                      >
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-medium text-foreground capitalize">{key}</span>
                          {!upstreamHas && <Badge variant="outline">not advertised upstream</Badge>}
                        </div>
                        <Switch
                          checked={Boolean(selectedCapabilities[key])}
                          onCheckedChange={(checked) =>
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
              </CollapsibleContent>
            </Collapsible>
          </div>
        </div>
      )}
    </div>
  );
};

export default AgentCardDiscovery;
