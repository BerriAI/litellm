import React from "react";
import { Table, TableHead, TableRow, TableHeaderCell, TableBody, TableCell, Button } from "@tremor/react";
import { Tooltip } from "antd";
import { Agent } from "./types";

interface AgentTableProps {
  agentsList: Agent[];
  isLoading: boolean;
  onDeleteClick: (agentId: string, agentName: string) => void;
  accessToken: string | null;
  onAgentUpdated: () => void;
  isAdmin: boolean;
  onAgentClick: (agentId: string) => void;
}

const AgentTable: React.FC<AgentTableProps> = ({
  agentsList,
  isLoading,
  onDeleteClick,
  accessToken,
  onAgentUpdated,
  isAdmin,
  onAgentClick,
}) => {
  if (isLoading) {
    return <div>Loading agents...</div>;
  }

  if (!agentsList || agentsList.length === 0) {
    return <div>No agents found. Create one to get started.</div>;
  }

  return (
    <Table>
      <TableHead>
        <TableRow>
          <TableHeaderCell>Agent Name</TableHeaderCell>
          <TableHeaderCell>Description</TableHeaderCell>
          <TableHeaderCell>Created At</TableHeaderCell>
          {isAdmin && <TableHeaderCell>Actions</TableHeaderCell>}
        </TableRow>
      </TableHead>
      <TableBody>
        {agentsList.map((agent) => (
          <TableRow key={agent.agent_id}>
            <TableCell>
              <Tooltip title={agent.agent_name || ""}>
                <Button
                  size="xs"
                  variant="light"
                  className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left overflow-hidden truncate max-w-[200px]"
                  onClick={() => onAgentClick(agent.agent_id)}
                >
                  {agent.agent_name || ""}
                </Button>
              </Tooltip>
            </TableCell>
            <TableCell>
              {agent.agent_card_params?.description || "No description"}
            </TableCell>
            <TableCell>
              {agent.created_at
                ? new Date(agent.created_at).toLocaleDateString()
                : "N/A"}
            </TableCell>
            {isAdmin && (
              <TableCell>
                <Button
                  size="xs"
                  color="red"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteClick(agent.agent_id, agent.agent_name);
                  }}
                >
                  Delete
                </Button>
              </TableCell>
            )}
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
};

export default AgentTable;

