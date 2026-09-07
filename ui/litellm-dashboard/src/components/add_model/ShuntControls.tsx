import { Info } from "lucide-react";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { Input } from "@/components/ui/input";
import React from "react";
import { ModelGroup } from "@/components/llm_calls/fetch_models";
import { AutoRouterShuntState, DEFAULT_AUTO_ROUTER_SHUNT_MIN_LINES } from "./buildAutoRouterShunt";

interface ShuntControlsProps {
  value: AutoRouterShuntState;
  onChange: (state: AutoRouterShuntState) => void;
  modelInfo: ModelGroup[];
}

const ShuntControls: React.FC<ShuntControlsProps> = ({ value, onChange, modelInfo }) => {
  const { minLines, bulkReadModel, codeWriteModel } = value;
  const armed = minLines !== undefined;

  const chatModelOptions = Array.from(
    new Set(
      modelInfo.filter((model) => model.mode === undefined || model.mode === "chat").map((model) => model.model_group),
    ),
  ).map((model_group) => ({ value: model_group, label: model_group }));

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <span className="text-sm font-medium">Bounded-read threshold</span>
        <SimpleTooltip content="Files this router's own Read/Bash tool calls read above this many lines are delegated to a cheaper model instead of entering the routed model's context.">
          <Info className="size-4 text-muted-foreground" />
        </SimpleTooltip>
      </div>
      <Input
        type="number"
        value={minLines ?? ""}
        onChange={(event) =>
          onChange({
            ...value,
            minLines: event.target.value === "" ? undefined : event.target.valueAsNumber,
          })
        }
        placeholder={String(DEFAULT_AUTO_ROUTER_SHUNT_MIN_LINES)}
        min={1}
        aria-label="Bounded-read threshold"
        className="w-full"
      />

      {armed && (
        <div className="grid gap-4 md:grid-cols-2 mt-4 pt-4 border-t border-border">
          <div>
            <span className="mb-1 block text-sm font-medium">Bulk-read model</span>
            <SearchSelect
              options={chatModelOptions}
              value={bulkReadModel ?? ""}
              onValueChange={(model) => onChange({ ...value, bulkReadModel: model === "" ? undefined : model })}
              placeholder="Use this router's default model"
              emptyText="No chat models found"
              aria-label="Bulk-read model"
            />
          </div>
          <div>
            <span className="mb-1 block text-sm font-medium">Code-write model</span>
            <SearchSelect
              options={chatModelOptions}
              value={codeWriteModel ?? ""}
              onValueChange={(model) => onChange({ ...value, codeWriteModel: model === "" ? undefined : model })}
              placeholder="Use this router's default model"
              emptyText="No chat models found"
              aria-label="Code-write model"
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default ShuntControls;
