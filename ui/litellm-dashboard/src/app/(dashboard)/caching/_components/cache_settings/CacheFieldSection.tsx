import React from "react";
import CacheFormField, { EmbeddingModelOption } from "./CacheFormField";
import { fieldsForSection } from "./cacheSettingsUtils";
import { CacheSection, RedisType } from "./cacheSettingsFields";

interface CacheFieldSectionProps {
  title: string;
  section: CacheSection;
  redisType: RedisType;
  embeddingModels: EmbeddingModelOption[];
  gridCols?: string;
  headingLevel?: "h4" | "h5";
  configuredSecrets?: ReadonlySet<string>;
}

const CacheFieldSection: React.FC<CacheFieldSectionProps> = ({
  title,
  section,
  redisType,
  embeddingModels,
  gridCols = "grid-cols-1 gap-6 sm:grid-cols-2",
  headingLevel = "h4",
  configuredSecrets,
}) => {
  const fields = fieldsForSection(section, redisType);
  if (fields.length === 0) {
    return null;
  }

  const Heading = headingLevel;

  return (
    <div className="space-y-6">
      <Heading className="text-sm font-medium text-gray-900">{title}</Heading>
      <div className={`grid ${gridCols}`}>
        {fields.map((field) => (
          <CacheFormField
            key={field.name}
            field={field}
            embeddingModels={embeddingModels}
            isSecretConfigured={configuredSecrets?.has(field.name) ?? false}
          />
        ))}
      </div>
    </div>
  );
};

export default CacheFieldSection;
