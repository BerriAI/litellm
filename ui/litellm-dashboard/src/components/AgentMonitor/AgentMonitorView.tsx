import { ShieldCheck, Clock, Download, ChevronDown } from "lucide-react";
import React, { useState } from "react";
import { ActiveAlerts } from "./ActiveAlerts";
import { AgentDetail } from "./AgentDetail";
import { AgentHealthChart } from "./AgentHealthChart";
import { AgentTable } from "./AgentTable";
import type { AgentData } from "./AgentTable";
import { StatsCards } from "./StatsCards";

interface AgentMonitorViewProps {
  accessToken?: string | null;
}

export default function AgentMonitorView({ accessToken = null }: AgentMonitorViewProps) {
  const [selectedAgent, setSelectedAgent] = useState<AgentData | null>(null);

  if (selectedAgent) {
    return (
      <div className="p-6 w-full min-w-0 flex-1">
        <AgentDetail agent={selectedAgent} onClose={() => setSelectedAgent(null)} accessToken={accessToken} />
      </div>
    );
  }

  return (
    <div className="p-6 w-full min-w-0 flex-1">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <div className="bg-blue-50 p-2 rounded-lg border border-blue-100">
            <ShieldCheck className="h-6 w-6 text-blue-600" />
          </div>
          <div>
            <h1 className="text-xl font-semibold text-gray-900">Agent Monitor</h1>
            <p className="text-sm text-gray-500 mt-0.5">
              Monitor agent performance across all requests
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          <div className="flex items-center px-3 py-2 bg-white border border-gray-200 rounded-md shadow-sm text-sm text-gray-700 hover:bg-gray-50 cursor-pointer transition-colors">
            <Clock className="h-4 w-4 text-gray-400 mr-2" />
            <span>4 Mar, 20:31 - 11 Mar, 20:31</span>
            <ChevronDown className="h-4 w-4 text-gray-400 ml-2" />
          </div>
          <button className="flex items-center px-3 py-2 bg-white border border-gray-200 rounded-md shadow-sm text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors">
            <Download className="h-4 w-4 text-gray-400 mr-2" />
            Export Data
          </button>
        </div>
      </div>

      <StatsCards />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        <div className="lg:col-span-2 h-[350px]">
          <AgentHealthChart />
        </div>
        <div className="h-[350px]">
          <ActiveAlerts />
        </div>
      </div>

      <AgentTable onSelectAgent={setSelectedAgent} />
    </div>
  );
}
