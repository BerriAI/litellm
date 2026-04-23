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
    <Card className="p-6">
      <h3 className="text-lg font-semibold text-foreground mb-2">Security</h3>
      <p className="text-muted-foreground mb-4">
        When enabled, requests to this endpoint will require a valid LiteLLM
        Virtual Key
      </p>
      {premiumUser ? (
        <Switch
          name="auth"
          checked={authEnabled}
          onCheckedChange={(checked) => onAuthChange(checked)}
        />
      ) : (
        <div>
          <div className="flex items-center mb-3 gap-2">
            <Switch disabled checked={false} />
            <span className="text-sm text-muted-foreground">
              Authentication (Premium)
            </span>
          </div>
          <div className="p-3 bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-900 rounded-lg">
            <p className="text-sm text-amber-800 dark:text-amber-200">
              Setting authentication for pass-through endpoints is a LiteLLM
              Enterprise feature. Get a trial key{" "}
              <a
                href="https://www.litellm.ai/#pricing"
                target="_blank"
                rel="noopener noreferrer"
                className="underline"
              >
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
