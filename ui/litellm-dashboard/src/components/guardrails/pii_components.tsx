import React from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { Ban, Eye, EyeOff, Filter, Info, X } from "lucide-react";
import { Select as AntSelect } from "antd";
import { PiiEntityCategory } from "./types";

const { Option } = AntSelect;

export const formatEntityName = (name: string) => {
  return name.replace(/_/g, " ");
};

export const getActionIcon = (action: string) => {
  switch (action) {
    case "MASK":
      return <EyeOff className="h-3.5 w-3.5 mr-1" />;
    case "BLOCK":
      return <Ban className="h-3.5 w-3.5 mr-1" />;
    default:
      return null;
  }
};

// CategoryFilter
export interface CategoryFilterProps {
  categories: PiiEntityCategory[];
  selectedCategories: string[];
  onChange: (categories: string[]) => void;
}

export const CategoryFilter: React.FC<CategoryFilterProps> = ({
  categories,
  selectedCategories,
  onChange,
}) => {
  return (
    <div>
      <div className="flex items-center mb-2">
        <Filter className="h-3.5 w-3.5 text-muted-foreground mr-1" />
        <span className="text-muted-foreground font-medium">
          Filter by category
        </span>
      </div>
      <AntSelect
        mode="multiple"
        placeholder="Select categories to filter by"
        style={{ width: "100%" }}
        onChange={onChange}
        value={selectedCategories}
        allowClear
        showSearch
        optionFilterProp="children"
        className="mb-4"
      >
        {categories.map((cat) => (
          <Option key={cat.category} value={cat.category}>
            {cat.category}
          </Option>
        ))}
      </AntSelect>
    </div>
  );
};

// QuickActions
export interface QuickActionsProps {
  onSelectAll: (action: string) => void;
  onUnselectAll: () => void;
  hasSelectedEntities: boolean;
}

export const QuickActions: React.FC<QuickActionsProps> = ({
  onSelectAll,
  onUnselectAll,
  hasSelectedEntities,
}) => {
  return (
    <div className="bg-muted p-5 rounded-lg mb-6 border border-border shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center">
          <span className="font-bold text-foreground text-base">
            Quick Actions
          </span>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Info className="ml-2 h-3 w-3 text-muted-foreground cursor-help" />
              </TooltipTrigger>
              <TooltipContent>
                Apply action to all PII types at once
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
        <Button
          variant="outline"
          onClick={onUnselectAll}
          disabled={!hasSelectedEntities}
          className="text-destructive border-destructive/30 hover:bg-destructive/10"
        >
          <X className="h-4 w-4" />
          Unselect All
        </Button>
      </div>
      <div className="grid grid-cols-2 gap-4">
        <Button
          variant="outline"
          onClick={() => onSelectAll("MASK")}
          className="h-10 text-primary border-primary/30 hover:bg-primary/10 w-full"
        >
          <EyeOff className="h-4 w-4" />
          Select All &amp; Mask
        </Button>
        <Button
          variant="outline"
          onClick={() => onSelectAll("BLOCK")}
          className="h-10 text-destructive border-destructive/30 hover:bg-destructive/10 w-full"
        >
          <Ban className="h-4 w-4" />
          Select All &amp; Block
        </Button>
      </div>
    </div>
  );
};

// PiiEntityList
export interface PiiEntityListProps {
  entities: string[];
  selectedEntities: string[];
  selectedActions: { [key: string]: string };
  actions: string[];
  onEntitySelect: (entity: string) => void;
  onActionSelect: (entity: string, action: string) => void;
  entityToCategoryMap: Map<string, string>;
}

export const PiiEntityList: React.FC<PiiEntityListProps> = ({
  entities,
  selectedEntities,
  selectedActions,
  actions,
  onEntitySelect,
  onActionSelect,
  entityToCategoryMap,
}) => {
  return (
    <div className="border border-border rounded-lg overflow-hidden shadow-sm">
      <div className="bg-muted px-5 py-3 border-b border-border flex">
        <span className="flex-1 text-foreground font-bold">PII Type</span>
        <span className="w-32 text-right text-foreground font-bold">Action</span>
      </div>
      <div className="max-h-[400px] overflow-y-auto">
        {entities.length === 0 ? (
          <div className="py-10 text-center text-muted-foreground">
            No PII types match your filter criteria
          </div>
        ) : (
          entities.map((entity) => {
            const isSelected = selectedEntities.includes(entity);
            return (
              <div
                key={entity}
                className={cn(
                  "px-5 py-3 flex items-center justify-between hover:bg-muted border-b border-border",
                  isSelected && "bg-blue-50 dark:bg-blue-950/30",
                )}
              >
                <label className="flex items-center flex-1 cursor-pointer">
                  <Checkbox
                    checked={isSelected}
                    onCheckedChange={() => onEntitySelect(entity)}
                    className="mr-3"
                  />
                  <span
                    className={cn(
                      isSelected
                        ? "font-medium text-foreground"
                        : "text-foreground/80",
                    )}
                  >
                    {formatEntityName(entity)}
                  </span>
                  {entityToCategoryMap.get(entity) && (
                    <Badge className="ml-2 text-xs bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300">
                      {entityToCategoryMap.get(entity)}
                    </Badge>
                  )}
                </label>
                <div className="w-32">
                  <Select
                    value={
                      isSelected ? selectedActions[entity] || "MASK" : "MASK"
                    }
                    onValueChange={(value) => onActionSelect(entity, value)}
                    disabled={!isSelected}
                  >
                    <SelectTrigger
                      className={cn(
                        "w-[120px] h-8",
                        !isSelected && "opacity-50",
                      )}
                    >
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {actions.map((action) => (
                        <SelectItem key={action} value={action}>
                          <div className="flex items-center">
                            {getActionIcon(action)}
                            {action}
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
