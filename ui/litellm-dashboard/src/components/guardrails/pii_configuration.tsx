import React, { useState } from "react";
import { Typography, Badge } from "antd";
import { PiiConfigurationProps } from "./types";
import { CategoryFilter, QuickActions, PiiEntityList } from "./pii_components";

const { Title, Text } = Typography;

/**
 * A reusable component for rendering PII entity selection and action configuration
 * Used in both add and edit guardrail forms
 */
const PiiConfiguration: React.FC<PiiConfigurationProps> = ({
  entities,
  actions,
  selectedEntities,
  selectedActions,
  onEntitySelect,
  onActionSelect,
  entityCategories = [],
}) => {
  const [selectedCategories, setSelectedCategories] = useState<string[]>([]);

  // Create a lookup map to quickly find an entity's category
  const entityToCategoryMap = new Map<string, string>();
  entityCategories.forEach((category) => {
    category.entities.forEach((entity) => {
      entityToCategoryMap.set(entity, category.category);
    });
  });

  // Filter entities based on selected categories
  const filteredEntities = entities.filter((entity) => {
    return selectedCategories.length === 0 || selectedCategories.includes(entityToCategoryMap.get(entity) || "");
  });

  // Select all entities with a specified action
  const handleSelectAll = (action: string) => {
    entities.forEach((entity) => {
      if (!selectedEntities.includes(entity)) {
        onEntitySelect(entity);
      }
      onActionSelect(entity, action);
    });
  };

  // Unselect all entities
  const handleUnselectAll = () => {
    // Instead of iterating through each entity and toggling,
    // we'll directly set the selected entities to an empty array
    // This is more reliable and ensures a clean slate
    selectedEntities.forEach((entity) => {
      onEntitySelect(entity);
    });
  };

  return (
    <div className="pii-configuration">
      <div className="flex justify-between items-center mb-5">
        <div className="flex items-center">
          <Title level={4} className="mb-0 font-semibold text-gray-800">
            Configure PII Protection
          </Title>
        </div>
        <Badge
          count={selectedEntities.length}
          showZero
          style={{ backgroundColor: selectedEntities.length > 0 ? "#4f46e5" : "#d9d9d9" }}
          overflowCount={999}
        >
          <Text className="text-gray-500">{selectedEntities.length} items selected</Text>
        </Badge>
      </div>

      <div className="mb-6">
        <CategoryFilter
          categories={entityCategories}
          selectedCategories={selectedCategories}
          onChange={setSelectedCategories}
        />

        <QuickActions
          onSelectAll={handleSelectAll}
          onUnselectAll={handleUnselectAll}
          hasSelectedEntities={selectedEntities.length > 0}
        />
      </div>

      <PiiEntityList
        entities={filteredEntities}
        selectedEntities={selectedEntities}
        selectedActions={selectedActions}
        actions={actions}
        onEntitySelect={onEntitySelect}
        onActionSelect={onActionSelect}
        entityToCategoryMap={entityToCategoryMap}
      />
    </div>
  );
};

export default PiiConfiguration;
