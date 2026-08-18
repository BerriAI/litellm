import { Settings, X } from "lucide-react";
import { useId, useState } from "react";
import { ComparisonInstance } from "../CompareUI";
import { MessageDisplay } from "./MessageDisplay";
import { UnifiedSelector } from "./UnifiedSelector";
import TagSelector from "@/components/tag_management/TagSelector";
import VectorStoreSelector from "@/components/vector_store_management/VectorStoreSelector";
import GuardrailSelector from "@/components/guardrails/GuardrailSelector";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";
import { SelectorOption, EndpointConfig, isAgentEndpoint, getComparisonSelection } from "../endpoint_config";

interface ComparisonPanelProps {
  comparison: ComparisonInstance;
  onUpdate: (
    updates: Partial<ComparisonInstance>,
    options?: { applyToAll?: boolean; keysToApply?: (keyof ComparisonInstance)[] },
  ) => void;
  onRemove: () => void;
  canRemove: boolean;
  selectorOptions: SelectorOption[];
  isLoadingOptions: boolean;
  endpointConfig: EndpointConfig;
  apiKey: string;
}
export function ComparisonPanel({
  comparison,
  onUpdate,
  onRemove,
  canRemove,
  selectorOptions,
  isLoadingOptions,
  endpointConfig,
  apiKey,
}: ComparisonPanelProps) {
  const isA2AMode = isAgentEndpoint(endpointConfig.id);
  const currentSelection = getComparisonSelection(comparison, endpointConfig.id);
  const [popoverVisible, setPopoverVisible] = useState(false);
  const syncId = useId();
  const advancedParamsId = useId();

  const handleSyncChange = (checked: boolean) => {
    if (checked) {
      onUpdate(
        {
          applyAcrossModels: true,
          temperature: comparison.temperature,
          maxTokens: comparison.maxTokens,
          tags: [...comparison.tags],
          vectorStores: [...comparison.vectorStores],
          guardrails: [...comparison.guardrails],
          useAdvancedParams: comparison.useAdvancedParams,
        },
        {
          applyToAll: true,
          keysToApply: ["temperature", "maxTokens", "tags", "vectorStores", "guardrails", "useAdvancedParams"],
        },
      );
    } else {
      // When unsyncing, just turn off the sync flag - don't reset values
      onUpdate({
        applyAcrossModels: false,
      });
    }
  };

  const handleAdvancedParamsChange = (checked: boolean) => {
    onUpdate(
      {
        useAdvancedParams: checked,
      },
      comparison.applyAcrossModels ? { applyToAll: true, keysToApply: ["useAdvancedParams"] } : undefined,
    );
  };

  const handleSettingChange = <K extends keyof ComparisonInstance>(key: K, value: ComparisonInstance[K]) => {
    onUpdate(
      {
        [key]: value,
      } as Partial<ComparisonInstance>,
      comparison.applyAcrossModels ? { applyToAll: true, keysToApply: [key] } : undefined,
    );
  };

  const disabledOpacity = comparison.useAdvancedParams ? 1 : 0.4;
  const disabledTextColor = comparison.useAdvancedParams ? "text-foreground" : "text-muted-foreground";

  const handleClosePopover = () => {
    setPopoverVisible(false);
  };

  const settingsContent = (
    <div className="relative max-h-[65vh] w-[300px] overflow-y-auto">
      <Button
        type="button"
        variant="ghost"
        size="icon-xs"
        className="absolute top-0 right-0 z-10 text-muted-foreground"
        aria-label="Close settings"
        onClick={handleClosePopover}
      >
        <X />
      </Button>

      <div className="space-y-2">
        {/* Sync Checkbox */}
        <div className="flex items-center gap-2">
          <Checkbox
            id={syncId}
            checked={comparison.applyAcrossModels}
            onCheckedChange={handleSyncChange}
            aria-label="Sync Settings Across Models"
          />
          <label htmlFor={syncId} className="cursor-pointer text-xs font-medium">
            Sync Settings Across Models
          </label>
        </div>

        <Separator className="my-3" />

        {/* General Settings */}
        <div>
          <h4 className="mb-1.5 text-[11px] font-semibold tracking-wide text-muted-foreground uppercase">
            General Settings
          </h4>
          <div className="space-y-2">
            <div>
              <label className="mb-0.5 block text-xs font-medium text-muted-foreground">Tags</label>
              <TagSelector
                value={comparison.tags}
                onChange={(value) => handleSettingChange("tags", value)}
                accessToken={apiKey}
              />
            </div>
            <div>
              <label className="mb-0.5 block text-xs font-medium text-muted-foreground">Vector Stores</label>
              <VectorStoreSelector
                value={comparison.vectorStores}
                onChange={(value) => handleSettingChange("vectorStores", value)}
                accessToken={apiKey}
              />
            </div>
            <div>
              <label className="mb-0.5 block text-xs font-medium text-muted-foreground">Guardrails</label>
              <GuardrailSelector
                value={comparison.guardrails}
                onChange={(value) => handleSettingChange("guardrails", value)}
                accessToken={apiKey}
              />
            </div>
          </div>
        </div>
        {/* Advanced Settings */}
        <div>
          <h4 className="mb-1.5 text-[11px] font-semibold tracking-wide text-muted-foreground uppercase">
            Advanced Settings
          </h4>
          <div className="space-y-2">
            <div className="flex items-center gap-2 pb-1">
              <Checkbox
                id={advancedParamsId}
                checked={comparison.useAdvancedParams}
                onCheckedChange={handleAdvancedParamsChange}
                aria-label="Use Advanced Parameters"
              />
              <label htmlFor={advancedParamsId} className="cursor-pointer text-sm font-medium">
                Use Advanced Parameters
              </label>
            </div>
            <div className="space-y-2 transition-opacity duration-200" style={{ opacity: disabledOpacity }}>
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className={`text-xs font-medium ${disabledTextColor}`}>Temperature</label>
                  <span className={`text-xs ${disabledTextColor}`}>{comparison.temperature.toFixed(2)}</span>
                </div>
                <Slider
                  min={0}
                  max={2}
                  step={0.01}
                  value={[comparison.temperature]}
                  onValueChange={(value) => {
                    const nextValue = Array.isArray(value) ? value[0] : value;
                    const clamped = Math.min(2, Math.max(0, Number(nextValue.toFixed(2))));
                    handleSettingChange("temperature", clamped);
                  }}
                  disabled={!comparison.useAdvancedParams}
                />
              </div>
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className={`text-xs font-medium ${disabledTextColor}`}>Max Tokens</label>
                  <span className={`text-xs ${disabledTextColor}`}>{comparison.maxTokens}</span>
                </div>
                <Slider
                  min={1}
                  max={32768}
                  step={1}
                  value={[comparison.maxTokens]}
                  onValueChange={(value) => {
                    const nextValue = Array.isArray(value) ? value[0] : value;
                    const clamped = Math.min(32768, Math.max(1, Math.round(nextValue)));
                    handleSettingChange("maxTokens", clamped);
                  }}
                  disabled={!comparison.useAdvancedParams}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="flex min-h-0 flex-col border-l border-border bg-background first:border-l-0">
      <div className="flex items-center justify-between gap-3 border-b px-4 py-3">
        <div className="flex flex-1 items-center gap-3">
          <UnifiedSelector
            value={currentSelection}
            options={selectorOptions}
            loading={isLoadingOptions}
            config={endpointConfig}
            onChange={(value) => onUpdate(isA2AMode ? { agent: value } : { model: value })}
          />
          <div className="flex items-center gap-2">
            <Popover open={popoverVisible} onOpenChange={setPopoverVisible}>
              <PopoverTrigger
                render={<Button type="button" variant="ghost" size="icon-sm" aria-label="Panel settings" />}
              >
                <Settings />
              </PopoverTrigger>
              <PopoverContent side="bottom" align="end" className="w-auto">
                {settingsContent}
              </PopoverContent>
            </Popover>
          </div>
        </div>
        {canRemove && (
          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            className="text-destructive hover:bg-destructive/10 hover:text-destructive"
            aria-label="Remove comparison"
            onClick={(event) => {
              event.stopPropagation();
              onRemove();
            }}
          >
            <X />
          </Button>
        )}
      </div>
      <div className="relative flex-1 flex flex-col min-h-0">
        <div className="flex-1 max-h-[calc(100vh-385px)] overflow-auto rounded-b-2xl">
          <MessageDisplay messages={comparison.messages} isLoading={comparison.isLoading} />
        </div>
      </div>
    </div>
  );
}
