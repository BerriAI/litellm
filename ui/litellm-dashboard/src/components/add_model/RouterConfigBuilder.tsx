import { Info, Plus, Trash2, X } from "lucide-react";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import React, { useEffect, useState } from "react";
import { ModelGroup } from "../playground/llm_calls/fetch_models";

const InfoTip: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <TooltipProvider>
    <Tooltip>
      <TooltipTrigger asChild>
        <Info className="h-4 w-4 text-muted-foreground" />
      </TooltipTrigger>
      <TooltipContent className="max-w-xs">{children}</TooltipContent>
    </Tooltip>
  </TooltipProvider>
);

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

function UtterancesInput({
  value,
  onChange,
}: {
  value: string[];
  onChange: (next: string[]) => void;
}) {
  const [draft, setDraft] = useState("");

  const addValues = (input: string) => {
    const parts = input
      .split(/[\n]/)
      .map((part) => part.trim())
      .filter((part) => part.length > 0);
    if (parts.length === 0) return;
    const next = [...value];
    for (const part of parts) {
      if (!next.includes(part)) {
        next.push(part);
      }
    }
    onChange(next);
  };

  return (
    <div className="space-y-2">
      <Input
        value={draft}
        placeholder="Type an utterance and press Enter..."
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            addValues(draft);
            setDraft("");
          }
        }}
        onPaste={(e) => {
          const text = e.clipboardData.getData("text");
          if (text.includes("\n")) {
            e.preventDefault();
            addValues(text);
            setDraft("");
          }
        }}
      />
      {value.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {value.map((u) => (
            <Badge key={u} variant="secondary" className="gap-1">
              <span>{u}</span>
              <button
                type="button"
                aria-label={`Remove ${u}`}
                onClick={() => onChange(value.filter((x) => x !== u))}
              >
                <X className="h-3 w-3" />
              </button>
            </Badge>
          ))}
        </div>
      )}
    </div>
  );
}

const RouterConfigBuilder: React.FC<RouterConfigBuilderProps> = ({ modelInfo, value, onChange }) => {
  const [routes, setRoutes] = useState<Route[]>([]);
  const [showJsonPreview, setShowJsonPreview] = useState<boolean>(false);
  const [expandedRoutes, setExpandedRoutes] = useState<string[]>([]);

  useEffect(() => {
    const routesFromValue = value?.routes;
    if (routesFromValue) {
      const routeIds: string[] = [];
      setRoutes((prevRoutes) => {
        const initializedRoutes = routesFromValue.map((route: SavedRoute, index: number) => {
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
        });
        return initializedRoutes;
      });
      setExpandedRoutes(routeIds);
    } else {
      setRoutes([]);
      setExpandedRoutes([]);
    }
  }, [value]);

  const addRoute = () => {
    const newRouteId = `route-${Date.now()}`;
    const newRoute: Route = {
      id: newRouteId,
      model: "",
      utterances: [],
      description: "",
      score_threshold: 0.5,
    };
    const updatedRoutes = [...routes, newRoute];
    setRoutes(updatedRoutes);
    updateConfig(updatedRoutes);
    setExpandedRoutes((prev) => [...prev, newRouteId]);
  };

  const removeRoute = (routeId: string) => {
    const updatedRoutes = routes.filter((route) => route.id !== routeId);
    setRoutes(updatedRoutes);
    updateConfig(updatedRoutes);
    setExpandedRoutes((prev) => prev.filter((id) => id !== routeId));
  };

  const updateRoute = (routeId: string, field: keyof Route, value: any) => {
    const updatedRoutes = routes.map((route) => (route.id === routeId ? { ...route, [field]: value } : route));
    setRoutes(updatedRoutes);
    updateConfig(updatedRoutes);
  };

  const updateConfig = (updatedRoutes: Route[]) => {
    const config = {
      routes: updatedRoutes.map((route) => ({
        name: route.model,
        utterances: route.utterances,
        description: route.description,
        score_threshold: route.score_threshold,
      })),
    };
    onChange?.(config);
  };

  const generateConfig = () => {
    return {
      routes: routes.map((route) => ({
        name: route.model,
        utterances: route.utterances,
        description: route.description,
        score_threshold: route.score_threshold,
      })),
    };
  };

  return (
    <div className="w-full max-w-none">
      <div className="flex justify-between items-center w-full mb-6">
        <div className="flex items-center gap-2">
          <h4 className="text-base font-semibold m-0">Routes Configuration</h4>
          <InfoTip>
            Configure routing logic to automatically select the best model
            based on user input patterns
          </InfoTip>
        </div>
        <Button type="button" onClick={addRoute}>
          <Plus className="h-4 w-4" />
          Add Route
        </Button>
      </div>

      {/* Routes */}
      {routes.length === 0 ? (
        <Card className="p-6">
          <div className="py-8 flex flex-col items-center justify-center text-muted-foreground">
            <div className="text-sm">
              No routes configured. Click &quot;Add Route&quot; to get started.
            </div>
          </div>
        </Card>
      ) : (
        <Accordion
          type="multiple"
          value={expandedRoutes}
          onValueChange={(next) => setExpandedRoutes(next)}
          className="w-full"
        >
          {routes.map((route, index) => (
            <AccordionItem key={route.id} value={route.id}>
              <div className="flex items-center gap-2">
                <AccordionTrigger className="flex-1">
                  <span className="text-base">
                    Route {index + 1}: {route.model || "Unnamed"}
                  </span>
                </AccordionTrigger>
                <Button
                  type="button"
                  size="icon"
                  variant="ghost"
                  className="h-7 w-7 text-destructive"
                  onClick={(e) => {
                    e.stopPropagation();
                    removeRoute(route.id);
                  }}
                  aria-label="Remove route"
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
              <AccordionContent>
                <Card key={route.id} className="p-4">
                  <div className="mb-4 w-full">
                    <span className="text-sm font-medium mb-2 block">Model</span>
                    <Select
                      value={route.model || undefined}
                      onValueChange={(value) => updateRoute(route.id, "model", value)}
                    >
                      <SelectTrigger className="w-full">
                        <SelectValue placeholder="Select model" />
                      </SelectTrigger>
                      <SelectContent>
                        {modelInfo.map((model) => (
                          <SelectItem key={model.model_group} value={model.model_group}>
                            {model.model_group}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="mb-4 w-full">
                    <span className="text-sm font-medium mb-2 block">Description</span>
                    <Textarea
                      value={route.description}
                      onChange={(e) => updateRoute(route.id, "description", e.target.value)}
                      placeholder="Describe when this route should be used..."
                      rows={2}
                      className="w-full"
                    />
                  </div>

                  <div className="mb-4 w-full">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-sm font-medium">Score Threshold</span>
                      <InfoTip>
                        Minimum similarity score to route to this model (0-1)
                      </InfoTip>
                    </div>
                    <Input
                      type="number"
                      value={route.score_threshold}
                      onChange={(e) => {
                        const raw = e.target.value;
                        if (raw === "") {
                          updateRoute(route.id, "score_threshold", 0);
                          return;
                        }
                        const num = Number(raw);
                        updateRoute(route.id, "score_threshold", Number.isNaN(num) ? 0 : num);
                      }}
                      min={0}
                      max={1}
                      step={0.1}
                      placeholder="0.5"
                      className="w-full"
                    />
                  </div>

                  <div className="w-full">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-sm font-medium">Example Utterances</span>
                      <InfoTip>
                        Training examples for this route. Type an utterance and press
                        Enter to add it.
                      </InfoTip>
                    </div>
                    <p className="text-xs text-muted-foreground mb-2">
                      Type an utterance and press Enter to add it. You can also paste
                      multiple lines.
                    </p>
                    <UtterancesInput
                      value={route.utterances}
                      onChange={(next) => updateRoute(route.id, "utterances", next)}
                    />
                  </div>
                </Card>
              </AccordionContent>
            </AccordionItem>
          ))}
        </Accordion>
      )}

      {/* JSON Preview */}
      <hr className="border-border my-6" />
      <div className="flex justify-between items-center mb-4 w-full">
        <span className="text-lg font-semibold">JSON Preview</span>
        <Button
          type="button"
          variant="link"
          onClick={() => setShowJsonPreview(!showJsonPreview)}
          className="p-0 h-auto"
        >
          {showJsonPreview ? "Hide" : "Show"}
        </Button>
      </div>

      {showJsonPreview && (
        <Card className="bg-muted w-full p-4">
          <pre className="text-sm overflow-auto max-h-64 w-full">
            {JSON.stringify(generateConfig(), null, 2)}
          </pre>
        </Card>
      )}
    </div>
  );
};

export default RouterConfigBuilder;
