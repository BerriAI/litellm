"use client";

import React from "react";
import { PiggyBank } from "lucide-react";
import { Alert, Tabs } from "antd";

import UsageTab from "./UsageTab";
import PromptCachingTab from "./PromptCachingTab";
import { useDailyActivityRange } from "./useDailyActivityRange";

interface CostOptimizationViewProps {
  accessToken: string | null;
  userId: string | null;
  userRole: string;
}

const CostOptimizationView: React.FC<CostOptimizationViewProps> = ({ accessToken, userId, userRole }) => {
  const activity = useDailyActivityRange(accessToken, userId, userRole);

  const items = [
    {
      key: "usage",
      label: "Usage",
      children: <UsageTab accessToken={accessToken} activity={activity} />,
    },
    {
      key: "caching",
      label: "Prompt Caching",
      children: <PromptCachingTab accessToken={accessToken} activity={activity} />,
    },
  ];

  return (
    <div className="w-full space-y-6 p-6">
      <div>
        <div className="flex items-center gap-2">
          <PiggyBank className="size-6 text-emerald-600" strokeWidth={1.75} />
          <h1 className="text-xl font-semibold text-foreground">Cost Optimization</h1>
        </div>
        <p className="mt-1 text-sm text-muted-foreground">
          Track what the money-saving mechanisms are actually saving you. Prompt compression is configured under AI
          Gateway, on the Compression page; auto routers live under Models + Endpoints, on the Auto-Routers tab
        </p>
      </div>

      <Alert
        type="info"
        showIcon
        message="This is an experimental dashboard"
        description={
          <span>
            Have feedback? Join the discussion{" "}
            <a
              href="https://github.com/BerriAI/litellm/discussions/32172"
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 underline"
            >
              here
            </a>
          </span>
        }
      />

      <Tabs defaultActiveKey="usage" items={items} />
    </div>
  );
};

export default CostOptimizationView;
