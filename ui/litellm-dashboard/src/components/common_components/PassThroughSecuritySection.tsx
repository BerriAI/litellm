import React from "react";

import { Card } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";

export interface PassThroughSecuritySectionProps {
  authEnabled: boolean;
  onAuthChange: (checked: boolean) => void;
}

const PassThroughSecuritySection: React.FC<PassThroughSecuritySectionProps> = ({ authEnabled, onAuthChange }) => {
  return (
    <Card className="block p-6">
      <h3 className="mb-2 text-lg font-semibold text-foreground">Security</h3>
      <p className="mb-4 text-sm text-muted-foreground">
        When enabled, requests to this endpoint will require a valid LiteLLM Virtual Key
      </p>
      <Switch aria-label="Require LiteLLM authentication" checked={authEnabled} onCheckedChange={onAuthChange} />
    </Card>
  );
};

export default PassThroughSecuritySection;
