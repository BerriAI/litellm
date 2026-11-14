import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { SettingsIcon } from "lucide-react";
import { useTranslation } from "react-i18next";
import ModelSelector from "@/components/common_components/ModelSelector";

interface ModelConfigCardProps {
  model: string | null;
  temperature?: number;
  maxTokens?: number;
  accessToken: string | null;
  onModelChange: (model: string | null) => void;
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
  const { t } = useTranslation();
  const [showConfig, setShowConfig] = useState(false);

  return (
    <div className="flex items-center gap-3">
      <div className="w-[300px]">
        <ModelSelector accessToken={accessToken || ""} value={model} onChange={onModelChange} showLabel={false} />
      </div>

      <Button type="button" variant="outline" onClick={() => setShowConfig(!showConfig)} className="gap-2">
        <SettingsIcon size={16} />
        <span>{t("prompts.modelConfig.parameters")}</span>
      </Button>

      <Dialog open={showConfig} onOpenChange={setShowConfig}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t("prompts.modelConfig.title")}</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <div>
              <div className="flex items-center justify-between mb-2">
                <label htmlFor="prompt-temperature" className="text-sm text-foreground">
                  {t("prompts.modelConfig.temperature")}
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
                  {t("prompts.modelConfig.maxTokens")}
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
