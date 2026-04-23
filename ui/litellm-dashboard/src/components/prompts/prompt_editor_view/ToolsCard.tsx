import React from "react";
import { Card } from "@/components/ui/card";
import { Plus, Trash } from "lucide-react";
import { Tool } from "./types";

interface ToolsCardProps {
  tools: Tool[];
  onAddTool: () => void;
  onEditTool: (index: number) => void;
  onRemoveTool: (index: number) => void;
}

const ToolsCard: React.FC<ToolsCardProps> = ({
  tools,
  onAddTool,
  onEditTool,
  onRemoveTool,
}) => {
  return (
    <Card className="p-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium">Tools</span>
        <button
          type="button"
          onClick={onAddTool}
          className="text-xs text-primary hover:underline flex items-center"
        >
          <Plus size={14} className="mr-1" />
          Add
        </button>
      </div>
      {tools.length === 0 ? (
        <p className="text-muted-foreground text-xs m-0">No tools added</p>
      ) : (
        <div className="space-y-2">
          {tools.map((tool, index) => (
            <div
              key={index}
              className="flex items-center justify-between p-2 bg-muted border border-border rounded"
            >
              <div className="flex-1 min-w-0">
                <div className="font-medium text-xs truncate">{tool.name}</div>
                <div className="text-xs text-muted-foreground truncate">
                  {tool.description}
                </div>
              </div>
              <div className="flex items-center space-x-1 ml-2">
                <button
                  type="button"
                  onClick={() => onEditTool(index)}
                  className="text-xs text-primary hover:underline"
                >
                  Edit
                </button>
                <button
                  type="button"
                  onClick={() => onRemoveTool(index)}
                  className="text-muted-foreground hover:text-destructive"
                  aria-label="Remove tool"
                >
                  <Trash size={14} />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </Card>
  );
};

export default ToolsCard;
