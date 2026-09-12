import React, { useEffect, useState } from "react";
import { ChevronDown, CircleHelp, Plus, Trash2, X } from "lucide-react";

import { ModelGroup } from "@/components/llm_calls/fetch_models";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

interface Route {
  id: string;
  model: string;
  utterances: string[];
  description: string;
  score_threshold: number;
}

interface SavedRoute {
  id?: string;
  name?: string;
  model?: string;
  utterances?: string[];
  description?: string;
  score_threshold?: number;
}

interface RouterConfig {
  routes?: SavedRoute[];
}

interface RouterConfigBuilderProps {
  modelInfo: ModelGroup[];
  value?: RouterConfig;
  onChange?: (config: any) => void;
}

interface UtteranceInputProps {
  value: string[];
  onChange: (utterances: string[]) => void;
}

const UtteranceInput = ({ value, onChange }: UtteranceInputProps) => {
  const [draft, setDraft] = useState("");

  const addUtterances = (input: string) => {
    const normalizedUtterances = input
      .split("\n")
      .map((utterance) => utterance.trim())
      .filter((utterance) => utterance !== "");
    const updatedUtterances = Array.from(new Set([...value, ...normalizedUtterances]));

    if (updatedUtterances.length > value.length) {
      onChange(updatedUtterances);
    }
    setDraft("");
  };

  return (
    <div className="flex min-h-9 w-full flex-wrap items-center gap-1.5 rounded-md border border-input bg-transparent px-2.5 py-1.5 shadow-xs transition-[color,box-shadow] focus-within:border-ring focus-within:ring-3 focus-within:ring-ring/50 dark:bg-input/30">
      {value.map((utterance) => (
        <Badge key={utterance} variant="secondary" className="max-w-full gap-1 pr-1">
          <span className="truncate">{utterance}</span>
          <button
            type="button"
            aria-label={`Remove ${utterance}`}
            className="rounded-full p-0.5 text-muted-foreground hover:bg-muted hover:text-foreground"
            onClick={() => onChange(value.filter((item) => item !== utterance))}
          >
            <X className="size-3" />
          </button>
        </Badge>
      ))}
      <input
        aria-label="Example Utterances"
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        onBlur={() => draft.trim() && addUtterances(draft)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && draft.trim()) {
            event.preventDefault();
            addUtterances(draft);
          } else if (event.key === "Backspace" && draft === "" && value.length > 0) {
            onChange(value.slice(0, -1));
          }
        }}
        onPaste={(event) => {
          const pastedText = event.clipboardData.getData("text");
          if (pastedText.includes("\n")) {
            event.preventDefault();
            addUtterances(pastedText);
          }
        }}
        placeholder={value.length === 0 ? "Type an utterance and press Enter..." : undefined}
        className="min-w-48 flex-1 bg-transparent py-0.5 text-sm outline-none placeholder:text-muted-foreground"
      />
    </div>
  );
};

const HelpTooltip = ({ content }: { content: string }) => (
  <Tooltip>
    <TooltipTrigger
      render={
        <button
          type="button"
          aria-label={content}
          className="inline-flex rounded-sm text-muted-foreground hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        />
      }
    >
      <CircleHelp className="size-4" />
    </TooltipTrigger>
    <TooltipContent>{content}</TooltipContent>
  </Tooltip>
);

const RouterConfigBuilder: React.FC<RouterConfigBuilderProps> = ({ modelInfo, value, onChange }) => {
  const [routes, setRoutes] = useState<Route[]>([]);
  const [showJsonPreview, setShowJsonPreview] = useState(false);
  const [expandedRoutes, setExpandedRoutes] = useState<string[]>([]);

  useEffect(() => {
    const routesFromValue = value?.routes;
    if (routesFromValue) {
      const routeIds: string[] = [];
      setRoutes((prevRoutes) =>
        routesFromValue.map((route: SavedRoute, index: number) => {
          const existingRoute = prevRoutes[index];
          const id = existingRoute?.id || route.id || `route-${index}-${Date.now()}`;
          routeIds.push(id);
          return {
            id,
            model: route.name || route.model || "",
            utterances: route.utterances || [],
            description: route.description || "",
            score_threshold: route.score_threshold ?? 0.5,
          };
        }),
      );
      setExpandedRoutes(routeIds);
    } else {
      setRoutes([]);
      setExpandedRoutes([]);
    }
  }, [value]);

  const updateConfig = (updatedRoutes: Route[]) => {
    onChange?.({
      routes: updatedRoutes.map((route) => ({
        name: route.model,
        utterances: route.utterances,
        description: route.description,
        score_threshold: route.score_threshold,
      })),
    });
  };

  const addRoute = () => {
    const newRouteId = `route-${Date.now()}`;
    const updatedRoutes = [
      ...routes,
      { id: newRouteId, model: "", utterances: [], description: "", score_threshold: 0.5 },
    ];
    setRoutes(updatedRoutes);
    updateConfig(updatedRoutes);
    setExpandedRoutes((previous) => [...previous, newRouteId]);
  };

  const removeRoute = (routeId: string) => {
    const updatedRoutes = routes.filter((route) => route.id !== routeId);
    setRoutes(updatedRoutes);
    updateConfig(updatedRoutes);
    setExpandedRoutes((previous) => previous.filter((id) => id !== routeId));
  };

  const updateRoute = (routeId: string, field: keyof Route, fieldValue: Route[keyof Route]) => {
    const updatedRoutes = routes.map((route) => (route.id === routeId ? { ...route, [field]: fieldValue } : route));
    setRoutes(updatedRoutes);
    updateConfig(updatedRoutes);
  };

  const modelOptions = modelInfo.map((model) => ({ value: model.model_group, label: model.model_group }));
  const generatedConfig = {
    routes: routes.map((route) => ({
      name: route.model,
      utterances: route.utterances,
      description: route.description,
      score_threshold: route.score_threshold,
    })),
  };

  return (
    <TooltipProvider>
      <div className="w-full space-y-6">
        <div className="flex w-full flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <h3 className="text-lg font-semibold">Routes Configuration</h3>
            <HelpTooltip content="Configure routing logic to automatically select the best model based on user input patterns" />
          </div>
          <Button type="button" onClick={addRoute}>
            <Plus data-icon="inline-start" />
            Add Route
          </Button>
        </div>

        {routes.length === 0 ? (
          <Card>
            <CardContent className="py-8 text-center text-muted-foreground">
              No routes configured. Click &quot;Add Route&quot; to get started.
            </CardContent>
          </Card>
        ) : (
          <div className="space-y-3">
            {routes.map((route, index) => {
              const isExpanded = expandedRoutes.includes(route.id);
              return (
                <Collapsible
                  key={route.id}
                  open={isExpanded}
                  onOpenChange={(open) =>
                    setExpandedRoutes((previous) =>
                      open ? [...previous, route.id] : previous.filter((id) => id !== route.id),
                    )
                  }
                  className="overflow-hidden rounded-xl border bg-card shadow-xs"
                >
                  <div className="flex items-center gap-2 px-4 py-3">
                    <CollapsibleTrigger
                      render={<button type="button" className="flex min-w-0 flex-1 items-center gap-2 text-left" />}
                    >
                      <ChevronDown
                        className={`size-4 shrink-0 text-muted-foreground transition-transform ${isExpanded ? "rotate-180" : ""}`}
                      />
                      <span className="truncate text-base font-medium">
                        Route {index + 1}: {route.model || "Unnamed"}
                      </span>
                    </CollapsibleTrigger>
                    <Button
                      type="button"
                      aria-label="delete"
                      variant="ghost"
                      size="icon-sm"
                      onClick={() => removeRoute(route.id)}
                    >
                      <Trash2 className="text-destructive" />
                    </Button>
                  </div>
                  <CollapsibleContent>
                    <Separator />
                    <div className="space-y-4 p-4">
                      <div className="space-y-2">
                        <Label>Model</Label>
                        <SearchSelect
                          value={route.model}
                          onValueChange={(model) => updateRoute(route.id, "model", model)}
                          placeholder="Select model"
                          options={modelOptions}
                        />
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor={`${route.id}-description`}>Description</Label>
                        <Textarea
                          id={`${route.id}-description`}
                          value={route.description}
                          onChange={(event) => updateRoute(route.id, "description", event.target.value)}
                          placeholder="Describe when this route should be used..."
                          rows={2}
                        />
                      </div>

                      <div className="space-y-2">
                        <div className="flex items-center gap-2">
                          <Label htmlFor={`${route.id}-threshold`}>Score Threshold</Label>
                          <HelpTooltip content="Minimum similarity score to route to this model (0-1)" />
                        </div>
                        <Input
                          id={`${route.id}-threshold`}
                          type="number"
                          value={route.score_threshold}
                          onChange={(event) =>
                            updateRoute(route.id, "score_threshold", Number(event.target.value) || 0)
                          }
                          min={0}
                          max={1}
                          step={0.1}
                          placeholder="0.5"
                        />
                      </div>

                      <div className="space-y-2">
                        <div className="flex items-center gap-2">
                          <Label>Example Utterances</Label>
                          <HelpTooltip content="Training examples for this route. Type an utterance and press Enter to add it." />
                        </div>
                        <p className="text-xs text-muted-foreground">
                          Type an utterance and press Enter to add it. You can also paste multiple lines.
                        </p>
                        <UtteranceInput
                          value={route.utterances}
                          onChange={(utterances) => updateRoute(route.id, "utterances", utterances)}
                        />
                      </div>
                    </div>
                  </CollapsibleContent>
                </Collapsible>
              );
            })}
          </div>
        )}

        <Separator />
        <div className="flex w-full items-center justify-between gap-3">
          <h3 className="text-lg font-semibold">JSON Preview</h3>
          <Button type="button" variant="link" onClick={() => setShowJsonPreview((visible) => !visible)}>
            {showJsonPreview ? "Hide" : "Show"}
          </Button>
        </div>

        {showJsonPreview && (
          <Card className="bg-muted/40">
            <CardContent>
              <pre className="max-h-64 w-full overflow-auto text-sm">{JSON.stringify(generatedConfig, null, 2)}</pre>
            </CardContent>
          </Card>
        )}
      </div>
    </TooltipProvider>
  );
};

export default RouterConfigBuilder;
