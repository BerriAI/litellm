import React from "react";
import CoordinationRedisFormField from "./CoordinationRedisFormField";
import { fieldsForSection } from "./coordinationRedisUtils";
import { CoordinationRedisType, CoordinationSection } from "./coordinationRedisFields";

interface CoordinationRedisFieldSectionProps {
  title: string;
  section: CoordinationSection;
  redisType: CoordinationRedisType;
  configuredSecrets: ReadonlySet<string>;
  gridCols?: string;
  headingLevel?: "h4" | "h5";
}

const CoordinationRedisFieldSection: React.FC<CoordinationRedisFieldSectionProps> = ({
  title,
  section,
  redisType,
  configuredSecrets,
  gridCols = "grid-cols-1 gap-6 sm:grid-cols-2",
  headingLevel = "h4",
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
          <CoordinationRedisFormField
            key={field.name}
            field={field}
            isSecretConfigured={configuredSecrets.has(field.name)}
          />
        ))}
      </div>
    </div>
  );
};

export default CoordinationRedisFieldSection;
