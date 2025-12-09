import React from "react";
import { Typography, Select, Button, Checkbox, Tooltip, Tag } from "antd";
import { CloseOutlined, EyeInvisibleOutlined, StopOutlined, FilterOutlined } from "@ant-design/icons";
import { PiiEntityCategory } from "./types";

const { Text } = Typography;
const { Option } = Select;

// Helper functions
export const formatEntityName = (name: string) => {
  return name.replace(/_/g, " ");
};

export const getActionIcon = (action: string) => {
  switch (action) {
    case "MASK":
      return <EyeInvisibleOutlined style={{ marginRight: 4 }} />;
    case "BLOCK":
      return <StopOutlined style={{ marginRight: 4 }} />;
    default:
      return null;
  }
};

// CategoryFilter component
export interface CategoryFilterProps {
  categories: PiiEntityCategory[];
  selectedCategories: string[];
  onChange: (categories: string[]) => void;
}

export const CategoryFilter: React.FC<CategoryFilterProps> = ({ categories, selectedCategories, onChange }) => {
  return (
    <div>
      <div className="flex items-center mb-2">
        <FilterOutlined className="text-gray-500 mr-1" />
        <Text className="text-gray-500 font-medium">Filter by category</Text>
      </div>
      <Select
        mode="multiple"
        placeholder="Select categories to filter by"
        style={{ width: "100%" }}
        onChange={onChange}
        value={selectedCategories}
        allowClear
        showSearch
        optionFilterProp="children"
        className="mb-4"
        tagRender={(props) => (
          <Tag color="blue" closable={props.closable} onClose={props.onClose} className="mr-2 mb-2">
            {props.label}
          </Tag>
        )}
      >
        {categories.map((cat) => (
          <Option key={cat.category} value={cat.category}>
            {cat.category}
          </Option>
        ))}
      </Select>
    </div>
  );
};

// QuickActions component
export interface QuickActionsProps {
  onSelectAll: (action: string) => void;
  onUnselectAll: () => void;
  hasSelectedEntities: boolean;
}

export const QuickActions: React.FC<QuickActionsProps> = ({ onSelectAll, onUnselectAll, hasSelectedEntities }) => {
  return (
    <div className="bg-gray-50 p-5 rounded-lg mb-6 border border-gray-200 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center">
          <Text strong className="text-gray-700 text-base">
            Quick Actions
          </Text>
          <Tooltip title="Apply action to all PII types at once">
            <div className="ml-2 text-gray-400 cursor-help text-xs">ⓘ</div>
          </Tooltip>
        </div>
        <Button
          type="default"
          danger
          onClick={onUnselectAll}
          disabled={!hasSelectedEntities}
          icon={<CloseOutlined />}
          className="border-gray-300 hover:text-red-600 hover:border-red-300"
        >
          Unselect All
        </Button>
      </div>
      <div className="grid grid-cols-2 gap-4">
        <Button
          type="default"
          onClick={() => onSelectAll("MASK")}
          className="flex items-center justify-center h-10 border-blue-200 hover:border-blue-300 hover:text-blue-700 bg-blue-50 hover:bg-blue-100 text-blue-600"
          block
          icon={<EyeInvisibleOutlined />}
        >
          Select All & Mask
        </Button>
        <Button
          type="default"
          danger
          onClick={() => onSelectAll("BLOCK")}
          className="flex items-center justify-center h-10 border-red-200 hover:border-red-300 hover:text-red-700 bg-red-50 hover:bg-red-100 text-red-600"
          block
          icon={<StopOutlined />}
        >
          Select All & Block
        </Button>
      </div>
    </div>
  );
};

// PiiEntityList component
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
    <div className="border rounded-lg overflow-hidden shadow-sm">
      <div className="bg-gray-50 px-5 py-3 border-b flex">
        <Text strong className="flex-1 text-gray-700">
          PII Type
        </Text>
        <Text strong className="w-32 text-right text-gray-700">
          Action
        </Text>
      </div>
      <div className="max-h-[400px] overflow-y-auto">
        {entities.length === 0 ? (
          <div className="py-10 text-center text-gray-500">No PII types match your filter criteria</div>
        ) : (
          entities.map((entity) => (
            <div
              key={entity}
              className={`px-5 py-3 flex items-center justify-between hover:bg-gray-50 border-b ${
                selectedEntities.includes(entity) ? "bg-blue-50" : ""
              }`}
            >
              <div className="flex items-center flex-1">
                <Checkbox
                  checked={selectedEntities.includes(entity)}
                  onChange={() => onEntitySelect(entity)}
                  className="mr-3"
                />
                <Text className={selectedEntities.includes(entity) ? "font-medium text-gray-900" : "text-gray-700"}>
                  {formatEntityName(entity)}
                </Text>
                {entityToCategoryMap.get(entity) && (
                  <Tag className="ml-2 text-xs" color="blue">
                    {entityToCategoryMap.get(entity)}
                  </Tag>
                )}
              </div>
              <div className="w-32">
                <Select
                  value={selectedEntities.includes(entity) ? selectedActions[entity] || "MASK" : "MASK"}
                  onChange={(value) => onActionSelect(entity, value)}
                  style={{ width: 120 }}
                  disabled={!selectedEntities.includes(entity)}
                  className={`${!selectedEntities.includes(entity) ? "opacity-50" : ""}`}
                  dropdownMatchSelectWidth={false}
                >
                  {actions.map((action) => (
                    <Option key={action} value={action}>
                      <div className="flex items-center">
                        {getActionIcon(action)}
                        {action}
                      </div>
                    </Option>
                  ))}
                </Select>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};
