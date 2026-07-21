import React from "react";
import CacheFormField, { EmbeddingModelOption } from "./CacheFormField";
import { fieldsForSection } from "./cacheSettingsUtils";
import { CacheSection, CacheType, RedisType } from "./cacheSettingsFields";

interface CacheFieldSectionProps {
  title: string;
  section: CacheSection;
  cacheType: CacheType;
  redisType: RedisType;
  embeddingModels: EmbeddingModelOption[];
  gridCols?: string;
  headingLevel?: "h4" | "h5";
}

const CacheFieldSection: React.FC<CacheFieldSectionProps> = ({
  title,
  section,
  cacheType,
  redisType,
  embeddingModels,
  gridCols = "grid-cols-1 gap-6 sm:grid-cols-2",
  headingLevel = "h4",
}) => {
  const fields = fieldsForSection(section, cacheType, redisType);
  if (fields.length === 0) {
    return null;
  }

  const Heading = headingLevel;

  return (
    <div className="space-y-6">
      <Heading className="text-sm font-medium text-gray-900">{title}</Heading>
      <div className={`grid ${gridCols}`}>
        {fields.map((field) => (
          <CacheFormField key={field.name} field={field} embeddingModels={embeddingModels} />
        ))}
      </div>
    </div>
  );
};

export default CacheFieldSection;
