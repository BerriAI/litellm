import {
  Card,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
  Title,
  Text,
} from "@tremor/react";
import { AlertTriangle, Settings } from "lucide-react";
import React, { useState } from "react";
import { Modal } from "antd";

export interface AgentData {
  id: string;
  name: string;
  type: string;
  driftScore: number;
  iterationsAvg: number;
  iterationsMax: number;
  status: "Healthy" | "Warning" | "Critical" | "Killed";
  killSwitchActive: boolean;
}

const mockData: AgentData[] = [
  { id: "1", name: "order-processor-v3", type: "Tool Agent", driftScore: 0.12, iterationsAvg: 4, iterationsMax: 8, status: "Healthy", killSwitchActive: false },
  { id: "2", name: "support-bot-v2", type: "Chat Agent", driftScore: 0.34, iterationsAvg: 6, iterationsMax: 15, status: "Warning", killSwitchActive: false },
  { id: "3", name: "data-analyst", type: "RAG Agent", driftScore: 0.05, iterationsAvg: 12, iterationsMax: 45, status: "Critical", killSwitchActive: false },
  { id: "4", name: "code-reviewer", type: "Code Agent", driftScore: 0.08, iterationsAvg: 3, iterationsMax: 5, status: "Healthy", killSwitchActive: false },
  { id: "5", name: "research-bot-v1", type: "Search Agent", driftScore: 0.67, iterationsAvg: 8, iterationsMax: 22, status: "Killed", killSwitchActive: true },
  { id: "6", name: "invoice-handler", type: "Workflow Agent", driftScore: 0.15, iterationsAvg: 2, iterationsMax: 4, status: "Warning", killSwitchActive: false },
  { id: "7", name: "content-writer", type: "Creative Agent", driftScore: 0.22, iterationsAvg: 5, iterationsMax: 7, status: "Healthy", killSwitchActive: false },
  { id: "8", name: "security-scanner", type: "Tool Agent", driftScore: 0.03, iterationsAvg: 1, iterationsMax: 2, status: "Healthy", killSwitchActive: false },
];

function getDriftColor(score: number): string {
  if (score < 0.2) return "text-emerald-600";
  if (score < 0.5) return "text-amber-500";
  return "text-red-500 font-semibold";
}

function StatusIndicator({ status }: { status: AgentData["status"] }) {
  const config = {
    Healthy: { dot: "bg-emerald-500", text: "text-gray-700", label: "Healthy", extra: "" },
    Warning: { dot: "bg-amber-500", text: "text-gray-700", label: "Warning", extra: "" },
    Critical: { dot: "bg-red-500 animate-pulse", text: "text-gray-700 font-medium", label: "Critical", extra: "" },
    Killed: { dot: "bg-gray-400", text: "text-gray-500 line-through", label: "Killed", extra: "" },
  }[status];

  return (
    <div className="flex items-center">
      <div className={`w-2 h-2 rounded-full ${config.dot} mr-2`} />
      <span className={`text-sm ${config.text}`}>{config.label}</span>
    </div>
  );
}

interface AgentTableProps {
  onSelectAgent?: (agent: AgentData) => void;
}

export const AgentTable: React.FC<AgentTableProps> = ({ onSelectAgent }) => {
  const [data, setData] = useState<AgentData[]>(mockData);
  const [confirmAgent, setConfirmAgent] = useState<AgentData | null>(null);

  const executeKillSwitch = (id: string) => {
    setData((prev) =>
      prev.map((agent) => {
        if (agent.id !== id) return agent;
        const newActive = !agent.killSwitchActive;
        return {
          ...agent,
          killSwitchActive: newActive,
          status: newActive
            ? "Killed"
            : agent.driftScore > 0.5
              ? "Critical"
              : agent.driftScore > 0.2
                ? "Warning"
                : "Healthy",
        };
      }),
    );
  };

  const handleKillSwitchClick = (agent: AgentData) => {
    if (agent.killSwitchActive) {
      executeKillSwitch(agent.id);
    } else {
      setConfirmAgent(agent);
    }
  };

  return (
    <Card className="bg-white border border-gray-200">
      <Modal
        title={
          <div className="flex items-center space-x-2">
            <AlertTriangle className="h-5 w-5 text-red-500" />
            <span>Activate Kill Switch</span>
          </div>
        }
        open={confirmAgent !== null}
        onCancel={() => setConfirmAgent(null)}
        onOk={() => {
          if (confirmAgent) executeKillSwitch(confirmAgent.id);
          setConfirmAgent(null);
        }}
        okText="Activate Kill Switch"
        okButtonProps={{ danger: true }}
        cancelText="Cancel"
      >
        <div className="py-2 space-y-3">
          <p className="text-sm text-gray-700">
            You are about to activate the kill switch for{" "}
            <span className="font-semibold font-mono">{confirmAgent?.name}</span>.
          </p>
          <div className="bg-red-50 border border-red-200 rounded-md p-3">
            <p className="text-sm text-red-800 font-medium mb-1">This will immediately:</p>
            <ul className="text-sm text-red-700 list-disc list-inside space-y-1">
              <li>Stop the agent from accepting any incoming requests</li>
              <li>Terminate all in-progress agent runs</li>
              <li>Mark the agent status as &ldquo;Killed&rdquo;</li>
            </ul>
          </div>
          <p className="text-xs text-gray-500">
            You can re-enable the agent later by toggling the kill switch off.
          </p>
        </div>
      </Modal>
      <div className="flex justify-between items-center mb-4">
        <div>
          <Title className="text-base font-semibold text-gray-900">Agent Status</Title>
          <Text className="text-sm text-gray-500 mt-0.5">
            Click an agent to view details and configuration
          </Text>
        </div>
        <button className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-50 rounded-md transition-colors border border-gray-200">
          <Settings className="h-4 w-4" />
        </button>
      </div>

      <Table>
        <TableHead>
          <TableRow>
            <TableHeaderCell>Agent</TableHeaderCell>
            <TableHeaderCell>Provider / Type</TableHeaderCell>
            <TableHeaderCell>Drift Score</TableHeaderCell>
            <TableHeaderCell>Iterations (Avg/Max)</TableHeaderCell>
            <TableHeaderCell>Status</TableHeaderCell>
            <TableHeaderCell className="text-right">Kill Switch</TableHeaderCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {data.map((agent) => (
            <TableRow
              key={agent.id}
              className="hover:bg-gray-50 cursor-pointer"
              onClick={() => onSelectAgent?.(agent)}
            >
              <TableCell>
                <span className={`text-sm font-medium ${agent.status === "Killed" ? "text-gray-400 line-through" : "text-gray-900"}`}>
                  {agent.name}
                </span>
              </TableCell>
              <TableCell>
                <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-mono font-medium bg-gray-100 text-gray-800 border border-gray-200">
                  {agent.type}
                </span>
              </TableCell>
              <TableCell>
                <span className={`text-sm ${getDriftColor(agent.driftScore)}`}>
                  {agent.driftScore.toFixed(2)}
                </span>
              </TableCell>
              <TableCell>
                <span className="text-sm text-gray-900">
                  {agent.iterationsAvg} <span className="text-gray-400">/</span> {agent.iterationsMax}
                </span>
              </TableCell>
              <TableCell>
                <StatusIndicator status={agent.status} />
              </TableCell>
              <TableCell className="text-right">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleKillSwitchClick(agent);
                  }}
                  className={`relative inline-flex h-5 w-9 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${agent.killSwitchActive ? "bg-red-500" : "bg-gray-200"}`}
                  role="switch"
                  aria-checked={agent.killSwitchActive}
                >
                  <span
                    aria-hidden="true"
                    className={`pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${agent.killSwitchActive ? "translate-x-4" : "translate-x-0"}`}
                  />
                </button>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Card>
  );
};
