import React from "react";
import { PlusIcon, TrashIcon } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Tool } from "./types";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

interface ToolsCardProps {
  tools: Tool[];
  onAddTool: () => void;
  onEditTool: (index: number) => void;
  onRemoveTool: (index: number) => void;
}

const ToolsCard: React.FC<ToolsCardProps> = ({ tools, onAddTool, onEditTool, onRemoveTool }) => {
  const { t } = useTranslation();
  return (
    <Card className="p-3">
      <div className="flex items-center justify-between mb-2">
        <p className="text-sm font-medium">{t("prompts.toolsCard.title")}</p>
        <Button variant="ghost" size="sm" onClick={onAddTool}>
          <PlusIcon size={14} className="mr-1" />
          {t("prompts.toolsCard.add")}
        </Button>
      </div>
      {tools.length === 0 ? (
        <p className="text-muted-foreground text-xs">{t("prompts.toolsCard.empty")}</p>
      ) : (
        <div className="space-y-2">
          {tools.map((tool, index) => (
            <div key={index} className="flex items-center justify-between p-2 bg-muted border border-border rounded-sm">
              <div className="flex-1 min-w-0">
                <div className="font-medium text-xs truncate">{tool.name}</div>
                <div className="text-xs text-muted-foreground truncate">{tool.description}</div>
              </div>
              <div className="flex items-center space-x-1 ml-2">
                <Button variant="ghost" size="sm" onClick={() => onEditTool(index)}>
                  {t("prompts.toolsCard.edit")}
                </Button>
                <Button
                  variant="ghost"
                  size="icon-sm"
                  aria-label={t("prompts.toolsCard.remove", { name: tool.name })}
                  onClick={() => onRemoveTool(index)}
                >
                  <TrashIcon size={14} aria-hidden="true" />
                </Button>
              </div>
            </div>
          ))}
        </div>
      )}
    </Card>
  );
};

export default ToolsCard;
