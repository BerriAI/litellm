import React, { useState } from "react";
import { CategoryFilter, PiiEntityList, QuickActions } from "./pii_components";
import { PiiConfigurationProps } from "@/components/guardrails/types";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { mergeCustomEntities, normalizeCustomEntityName } from "./piiCustomEntity";

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
  const [customEntityName, setCustomEntityName] = useState("");
  const [customEntityError, setCustomEntityError] = useState<string | null>(null);
  const allEntities = mergeCustomEntities(entities, selectedEntities);
  const customEntities = allEntities.filter((entity) => !entities.includes(entity));
  const categories =
    customEntities.length > 0
      ? [...entityCategories, { category: "Custom", entities: customEntities }]
      : entityCategories;

  // Create a lookup map to quickly find an entity's category
  const entityToCategoryMap = new Map<string, string>();
  entityCategories.forEach((category) => {
    category.entities.forEach((entity) => {
      entityToCategoryMap.set(entity, category.category);
    });
  });
  customEntities.forEach((entity) => {
    entityToCategoryMap.set(entity, "Custom");
  });

  // Filter entities based on selected categories
  const filteredEntities = allEntities.filter((entity) => {
    return selectedCategories.length === 0 || selectedCategories.includes(entityToCategoryMap.get(entity) || "");
  });

  // Select all entities with a specified action
  const handleSelectAll = (action: string) => {
    allEntities.forEach((entity) => {
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

  const handleAddCustomEntity = () => {
    const normalizedEntityName = normalizeCustomEntityName(customEntityName);
    if (normalizedEntityName === null) {
      setCustomEntityError("Use letters, numbers and underscores only");
      return;
    }

    setCustomEntityError(null);
    setCustomEntityName("");
    if (allEntities.includes(normalizedEntityName)) {
      return;
    }

    if (selectedCategories.length > 0 && !selectedCategories.includes("Custom")) {
      setSelectedCategories([...selectedCategories, "Custom"]);
    }
    onEntitySelect(normalizedEntityName);
    onActionSelect(normalizedEntityName, "MASK");
  };

  return (
    <div className="pii-configuration">
      <div className="flex justify-between items-center mb-5">
        <div className="flex items-center">
          <h4 className="m-0 text-lg font-semibold text-foreground">Configure PII Protection</h4>
        </div>
        <span className="text-muted-foreground">{selectedEntities.length} items selected</span>
      </div>

      <div className="mb-6">
        <CategoryFilter
          categories={categories}
          selectedCategories={selectedCategories}
          onChange={setSelectedCategories}
        />

        <QuickActions
          onSelectAll={handleSelectAll}
          onUnselectAll={handleUnselectAll}
          hasSelectedEntities={selectedEntities.length > 0}
        />
        <div className="mt-4">
          <div className="flex items-center gap-2">
            <Input
              value={customEntityName}
              onChange={(event) => {
                setCustomEntityName(event.target.value);
                setCustomEntityError(null);
              }}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault();
                  handleAddCustomEntity();
                }
              }}
              placeholder="Custom entity name, e.g. SE_PERSONNUMMER"
              aria-label="Custom entity name"
            />
            <Button type="button" onClick={handleAddCustomEntity}>
              Add entity
            </Button>
          </div>
          {customEntityError && <p className="mt-1 text-sm text-destructive">{customEntityError}</p>}
          <p className="mt-1 text-sm text-muted-foreground">
            Add any entity your Presidio analyzer recognizes, including custom recognizers.
          </p>
        </div>
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
