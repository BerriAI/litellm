import React from "react";

import { Card } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";

export interface PassThroughSecuritySectionProps {
  premiumUser: boolean;
  authEnabled: boolean;
  onAuthChange: (checked: boolean) => void;
}

const PassThroughSecuritySection: React.FC<PassThroughSecuritySectionProps> = ({
  premiumUser,
  authEnabled,
  onAuthChange,
}) => {
  return (
    <Card className="block p-6">
      <h3 className="mb-2 text-lg font-semibold text-foreground">Security</h3>
      <p className="mb-4 text-sm text-muted-foreground">
        When enabled, requests to this endpoint will require a valid LiteLLM Virtual Key
      </p>
      {premiumUser ? (
        <Switch checked={authEnabled} onCheckedChange={onAuthChange} />
      ) : (
        <div>
          <div className="mb-3 flex items-center">
            <Switch disabled checked={false} />
            <span className="ml-2 text-sm text-muted-foreground">Authentication (Premium)</span>
          </div>
          <div className="rounded-lg border border-warning/20 bg-warning/10 p-3">
            <p className="text-sm text-warning">
              Setting authentication for pass-through endpoints is a LiteLLM Enterprise feature. Get a trial key{" "}
              <a href="https://www.litellm.ai/#pricing" target="_blank" rel="noopener noreferrer" className="underline">
                here
              </a>
              .
            </p>
          </div>
        </div>
      )}
    </Card>
  );
};

export default PassThroughSecuritySection;
