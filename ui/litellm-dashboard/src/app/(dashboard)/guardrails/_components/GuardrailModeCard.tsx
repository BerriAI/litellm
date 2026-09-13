import React from "react";

import { setGuardrailEnabledCall } from "@/components/networking";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { toast } from "@/lib/toast";

import { formatGuardrailMode } from "./guardrail_info_helpers";

interface GuardrailModeCardProps {
  guardrailId: string;
  mode: unknown;
  defaultOn: boolean;
  enabled: boolean;
  accessToken: string | null;
  isAdmin: boolean;
  onChanged: () => Promise<void>;
}

const GuardrailModeCard: React.FC<GuardrailModeCardProps> = ({
  guardrailId,
  mode,
  defaultOn,
  enabled,
  accessToken,
  isAdmin,
  onChanged,
}) => {
  const handleToggle = async (nextEnabled: boolean) => {
    if (!accessToken) return;
    try {
      await setGuardrailEnabledCall(accessToken, guardrailId, nextEnabled);
      toast.success(`Guardrail ${nextEnabled ? "enabled" : "disabled"}`);
    } catch (error) {
      console.error("Error updating guardrail enabled state:", error);
      toast.fromError(`Failed to ${nextEnabled ? "enable" : "disable"} guardrail`);
    }
    await onChanged();
  };

  return (
    <Card className="block p-6">
      <p>Mode</p>
      <div className="mt-2">
        <h3 className="text-lg font-medium">{formatGuardrailMode(mode) || "-"}</h3>
        <div className="mt-1 flex items-center gap-2">
          <Badge variant={defaultOn ? "secondary" : "outline"}>{defaultOn ? "Default On" : "Default Off"}</Badge>
          <Badge variant={enabled ? "secondary" : "outline"}>{enabled ? "Enabled" : "Disabled"}</Badge>
        </div>
      </div>
      {isAdmin && (
        <div className="mt-4 flex items-center gap-3 text-sm">
          <Switch checked={enabled} aria-label="Enable guardrail" onCheckedChange={handleToggle} />
          <span>{enabled ? "Guardrail runs on requests" : "Guardrail is skipped for every request"}</span>
        </div>
      )}
    </Card>
  );
};

export default GuardrailModeCard;
