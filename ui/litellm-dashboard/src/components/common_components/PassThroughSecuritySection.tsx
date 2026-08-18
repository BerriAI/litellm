import React from "react";
import { Card, Title, Subtitle, Text } from "@tremor/react";

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
      <Title className="mb-2 text-lg font-semibold text-foreground">Security</Title>
      <Subtitle className="mb-4 text-muted-foreground">
        When enabled, requests to this endpoint will require a valid LiteLLM Virtual Key
      </Subtitle>
      {premiumUser ? (
        <Switch checked={authEnabled} onCheckedChange={onAuthChange} />
      ) : (
        <div>
          <div className="mb-3 flex items-center">
            <Switch disabled checked={false} />
            <span className="ml-2 text-sm text-muted-foreground">Authentication (Premium)</span>
          </div>
          <div className="rounded-lg border border-yellow-200 bg-yellow-50 p-3 dark:border-yellow-900 dark:bg-yellow-950">
            <Text className="text-sm text-yellow-800 dark:text-yellow-300">
              Setting authentication for pass-through endpoints is a LiteLLM Enterprise feature. Get a trial key{" "}
              <a href="https://www.litellm.ai/#pricing" target="_blank" rel="noopener noreferrer" className="underline">
                here
              </a>
              .
            </Text>
          </div>
        </div>
      )}
    </Card>
  );
};

export default PassThroughSecuritySection;
