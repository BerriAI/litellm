import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { SettingsIcon } from "lucide-react";
import ModelSelector from "@/components/common_components/ModelSelector";

interface ModelConfigCardProps {
  model: string;
  temperature?: number;
  maxTokens?: number;
  accessToken: string | null;
  onModelChange: (model: string) => void;
  onTemperatureChange: (temp: number) => void;
  onMaxTokensChange: (tokens: number) => void;
}

const ModelConfigCard: React.FC<ModelConfigCardProps> = ({
  model,
  temperature = 1,
  maxTokens = 1000,
  accessToken,
  onModelChange,
  onTemperatureChange,
  onMaxTokensChange,
}) => {
  const [showConfig, setShowConfig] = useState(false);

  return (
    <div className="flex items-center gap-3">
      <div className="w-[300px]">
        <ModelSelector accessToken={accessToken || ""} value={model} onChange={onModelChange} showLabel={false} />
      </div>

      <Button type="button" variant="outline" onClick={() => setShowConfig(!showConfig)} className="gap-2">
        <SettingsIcon size={16} />
        <span>Parameters</span>
      </Button>

      <Dialog open={showConfig} onOpenChange={setShowConfig}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Model Parameters</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <div>
              <div className="flex items-center justify-between mb-2">
                <label htmlFor="prompt-temperature" className="text-sm text-foreground">
                  Temperature
                </label>
                <Input
                  id="prompt-temperature"
                  type="number"
                  min={0}
                  max={2}
                  step={0.1}
                  value={temperature}
                  onChange={(e) => onTemperatureChange(parseFloat(e.target.value) || 0)}
                  className="w-20"
                />
              </div>
            </div>
            <div>
              <div className="flex items-center justify-between mb-2">
                <label htmlFor="prompt-max-tokens" className="text-sm text-foreground">
                  Max Tokens
                </label>
                <Input
                  id="prompt-max-tokens"
                  type="number"
                  min={1}
                  max={32768}
                  value={maxTokens}
                  onChange={(e) => onMaxTokensChange(parseInt(e.target.value) || 1000)}
                  className="w-24"
                />
              </div>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default ModelConfigCard;
