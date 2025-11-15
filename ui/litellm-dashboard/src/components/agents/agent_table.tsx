import React from "react";
import { Table, TableHead, TableRow, TableHeaderCell, TableBody, TableCell, Button } from "@tremor/react";
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
          <TableRow key={agent.agent_id} className="hover:bg-gray-50 cursor-pointer">
            <TableCell onClick={() => onAgentClick(agent.agent_id)}>
              {agent.agent_name}
            </TableCell>
            <TableCell onClick={() => onAgentClick(agent.agent_id)}>
              {agent.agent_card_params?.description || "No description"}
            </TableCell>
            <TableCell onClick={() => onAgentClick(agent.agent_id)}>
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

