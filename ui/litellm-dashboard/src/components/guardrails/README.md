# Guardrail Configuration Components

This directory contains reusable React components for configuring various guardrail types in the LiteLLM dashboard.

## Azure Text Moderation Configuration

### Overview

The Azure Text Moderation configuration component allows users to configure Azure Content Safety settings for text moderation. It provides an intuitive interface for:

- **Category Selection**: Choose which content categories to monitor (Hate, Sexual, SelfHarm, Violence)
- **Global Severity Threshold**: Set a default severity level threshold for all categories
- **Per-Category Thresholds**: Override the global threshold for specific categories (advanced)

### Components

#### `AzureTextModerationConfiguration`

The main configuration component that renders the complete UI for Azure text moderation settings.

**Props:**

```typescript
interface AzureTextModerationConfigurationProps {
  categories: string[]; // Available categories
  selectedCategories: string[]; // Currently selected categories
  globalSeverityThreshold: number; // Global threshold (0, 2, 4, 6)
  categorySpecificThresholds: { [key: string]: number }; // Per-category overrides
  onCategorySelect: (category: string) => void; // Category selection handler
  onGlobalSeverityChange: (threshold: number) => void; // Global threshold handler
  onCategorySeverityChange: (category: string, threshold: number) => void; // Per-category handler
}
```

#### `AzureTextModerationExample`

A complete example showing how to integrate the configuration component with state management.

### Usage Example

```tsx
import React, { useState } from "react";
import { AzureTextModerationConfiguration, AZURE_TEXT_MODERATION_CATEGORIES } from "./guardrails";

const MyComponent = () => {
  const [selectedCategories, setSelectedCategories] = useState<string[]>(["Hate", "Violence"]);
  const [globalSeverityThreshold, setGlobalSeverityThreshold] = useState<number>(2);
  const [categorySpecificThresholds, setCategorySpecificThresholds] = useState<{ [key: string]: number }>({});

  const handleCategorySelect = (category: string) => {
    setSelectedCategories((prev) =>
      prev.includes(category) ? prev.filter((c) => c !== category) : [...prev, category],
    );
  };

  const handleSave = () => {
    const config = {
      categories: selectedCategories,
      severity_threshold: globalSeverityThreshold,
      severity_threshold_by_category: categorySpecificThresholds,
    };
    // Save to backend...
  };

  return (
    <AzureTextModerationConfiguration
      categories={AZURE_TEXT_MODERATION_CATEGORIES.map((c) => c.name)}
      selectedCategories={selectedCategories}
      globalSeverityThreshold={globalSeverityThreshold}
      categorySpecificThresholds={categorySpecificThresholds}
      onCategorySelect={handleCategorySelect}
      onGlobalSeverityChange={setGlobalSeverityThreshold}
      onCategorySeverityChange={(category, threshold) =>
        setCategorySpecificThresholds((prev) => ({ ...prev, [category]: threshold }))
      }
    />
  );
};
```

### Severity Levels

The component supports Azure's severity levels:

- **Level 0 (Safe)**: Content is appropriate and safe
- **Level 2 (Low)**: Content may be inappropriate in some contexts
- **Level 4 (Medium)**: Content is inappropriate and should be filtered
- **Level 6 (High)**: Content is harmful and should be blocked

### Content Categories

Four predefined categories are supported:

1. **Hate**: Content that attacks or uses discriminatory language based on protected characteristics
2. **Sexual**: Content that describes sexual activity or other sexual content
3. **SelfHarm**: Content that promotes, encourages, or depicts acts of self-harm
4. **Violence**: Content that depicts death, violence, or physical injury

### Configuration Output

The component generates configuration objects compatible with the Azure Text Moderation guardrail:

```json
{
  "categories": ["Hate", "Violence"],
  "severity_threshold": 2,
  "severity_threshold_by_category": {
    "Hate": 4,
    "Violence": 2
  }
}
```

## PII Configuration

The existing PII configuration component provides similar functionality for configuring PII entity detection and actions (MASK/BLOCK).

### Files

- `azure_text_moderation_types.ts` - TypeScript interfaces and constants
- `azure_text_moderation_configuration.tsx` - Main configuration component
- `azure_text_moderation_example.tsx` - Usage example
- `pii_configuration.tsx` - PII configuration component
- `pii_components.tsx` - PII UI components
- `types.ts` - PII TypeScript interfaces
