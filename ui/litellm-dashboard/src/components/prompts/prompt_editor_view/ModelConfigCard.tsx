import React, { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Settings } from "lucide-react";
import ModelSelector from "../../common_components/ModelSelector";

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
        <ModelSelector
          accessToken={accessToken || ""}
          value={model}
          onChange={onModelChange}
          showLabel={false}
        />
      </div>

      <Button variant="outline" onClick={() => setShowConfig(!showConfig)}>
        <Settings className="h-4 w-4" />
        Parameters
      </Button>

      <Dialog open={showConfig} onOpenChange={setShowConfig}>
        <DialogContent className="max-w-96">
          <DialogHeader>
            <DialogTitle>Model Parameters</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-foreground">Temperature</span>
                <Input
                  type="number"
                  min={0}
                  max={2}
                  step={0.1}
                  value={temperature}
                  onChange={(e) =>
                    onTemperatureChange(parseFloat(e.target.value) || 0)
                  }
                  className="w-20 h-8"
                />
              </div>
            </div>
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-foreground">Max Tokens</span>
                <Input
                  type="number"
                  min={1}
                  max={32768}
                  value={maxTokens}
                  onChange={(e) =>
                    onMaxTokensChange(parseInt(e.target.value) || 1000)
                  }
                  className="w-24 h-8"
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
