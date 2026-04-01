import { Settings, X } from "lucide-react";
import { useState } from "react";
import { ComparisonInstance } from "../CompareUI";
import { MessageDisplay } from "./MessageDisplay";
import { UnifiedSelector } from "./UnifiedSelector";
import TagSelector from "../../../tag_management/TagSelector";
import VectorStoreSelector from "../../../vector_store_management/VectorStoreSelector";
import GuardrailSelector from "../../../guardrails/GuardrailSelector";
import { Checkbox, Divider, Popover, Slider } from "antd";
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
  const disabledTextColor = comparison.useAdvancedParams ? "text-gray-700" : "text-gray-400";

  const handleTogglePopover = () => {
    setPopoverVisible((prev) => !prev);
  };

  const handleClosePopover = () => {
    setPopoverVisible(false);
  };

  const settingsContent = (
    <div className="w-[300px] max-h-[65vh] overflow-y-auto relative">
      {/* Close button in top right */}
      <button
        onClick={handleClosePopover}
        className="absolute top-0 right-0 p-1 hover:bg-gray-100 rounded transition-colors text-gray-500 hover:text-gray-700 z-10"
      >
        <X size={14} />
      </button>

      <div className="space-y-2">
        {/* Sync Checkbox */}
        <div className="flex items-center gap-2">
          <Checkbox checked={comparison.applyAcrossModels} onChange={(e) => handleSyncChange(e.target.checked)}>
            <span className="text-xs font-medium">Sync Settings Across Models</span>
          </Checkbox>
        </div>

        <Divider className="border-gray-200" />

        {/* General Settings */}
        <div>
          <h4 className="text-xs font-semibold text-gray-700 mb-1.5 uppercase tracking-wide">General Settings</h4>
          <div className="space-y-2">
            <div>
              <label className="text-xs font-medium text-gray-600 block mb-0.5">Tags</label>
              <TagSelector
                value={comparison.tags}
                onChange={(value) => handleSettingChange("tags", value)}
                accessToken={apiKey}
              />
            </div>
            <div>
              <label className="text-xs font-medium text-gray-600 block mb-0.5">Vector Stores</label>
              <VectorStoreSelector
                value={comparison.vectorStores}
                onChange={(value) => handleSettingChange("vectorStores", value)}
                accessToken={apiKey}
              />
            </div>
            <div>
              <label className="text-xs font-medium text-gray-600 block mb-0.5">Guardrails</label>
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
          <h4 className="text-xs font-semibold text-gray-700 mb-1.5 uppercase tracking-wide">Advanced Settings</h4>
          <div className="space-y-2">
            <div className="flex items-center gap-2 pb-1">
              <Checkbox
                checked={comparison.useAdvancedParams}
                onChange={(e) => handleAdvancedParamsChange(e.target.checked)}
              >
                <span className="text-sm font-medium">Use Advanced Parameters</span>
              </Checkbox>
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
                  value={comparison.temperature}
                  onChange={(value) => {
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
                  value={comparison.maxTokens}
                  onChange={(value) => {
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
    <div className="bg-white first:border-l-0 border-l border-gray-200 flex flex-col min-h-0">
      <div className="border-b flex items-center justify-between gap-3 px-4 py-3">
        <div className="flex items-center gap-3 flex-1">
          <UnifiedSelector
            value={currentSelection}
            options={selectorOptions}
            loading={isLoadingOptions}
            config={endpointConfig}
            onChange={(value) =>
              onUpdate(isA2AMode ? { agent: value } : { model: value })
            }
          />
          <div className="flex items-center gap-2">
            <Popover
              content={settingsContent}
              trigger={[]}
              open={popoverVisible}
              onOpenChange={() => {
                // Prevent automatic closing - we control it manually
              }}
              placement="bottomRight"
              destroyTooltipOnHide={false}
            >
              <button
                onClick={(event) => {
                  event.stopPropagation();
                  handleTogglePopover();
                }}
                className={`p-2 rounded-lg transition-colors ${
                  popoverVisible ? "bg-gray-200 text-gray-700" : "hover:bg-gray-100 text-gray-600"
                }`}
              >
                <Settings size={18} />
              </button>
            </Popover>
          </div>
        </div>
        {canRemove && (
          <button
            onClick={(event) => {
              event.stopPropagation();
              onRemove();
            }}
            className="p-2 hover:bg-red-50 text-red-600 rounded-lg transition-colors"
          >
            <X size={18} />
          </button>
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
