"use client";

import React, { useState } from "react";
import { GuardrailGardenTab } from "./GuardrailGardenTab";
import { GuardrailsListTab } from "./GuardrailsListTab";
import { PlaygroundTab } from "./PlaygroundTab";
import { TeamGuardrailsTab } from "./TeamGuardrailsTab";

type Tab = "garden" | "guardrails" | "playground" | "team";

const TABS: { id: Tab; label: string }[] = [
  { id: "garden", label: "Guardrail Garden" },
  { id: "guardrails", label: "Guardrails" },
  { id: "playground", label: "Test Playground" },
  { id: "team", label: "Team Guardrails" },
];

interface GuardrailsPageProps {
  accessToken?: string | null;
}

export function GuardrailsPage({ accessToken }: GuardrailsPageProps) {
  const [activeTab, setActiveTab] = useState<Tab>("garden");

  return (
    <div className="flex flex-col w-full min-h-0 flex-1">
      {/* Tab bar */}
      <div className="border-b border-gray-200 px-6 flex-shrink-0 bg-white">
        <div className="flex items-center gap-0">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveTab(tab.id)}
              className={`relative px-4 py-3.5 text-sm font-medium transition-colors focus:outline-none ${
                activeTab === tab.id
                  ? "text-blue-500"
                  : "text-gray-500 hover:text-gray-700"
              } ${tab.id === "team" ? "flex items-center gap-1.5" : ""}`}
            >
              {tab.id === "team" && (
                <span className="inline-flex items-center justify-center w-1.5 h-1.5 rounded-full bg-blue-500" />
              )}
              {tab.label}
              {activeTab === tab.id && (
                <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-500 rounded-t-full" />
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-auto bg-white">
        {activeTab === "garden" && <GuardrailGardenTab />}
        {activeTab === "guardrails" && <GuardrailsListTab />}
        {activeTab === "playground" && <PlaygroundTab />}
        {activeTab === "team" && <TeamGuardrailsTab />}
      </div>
    </div>
  );
}
