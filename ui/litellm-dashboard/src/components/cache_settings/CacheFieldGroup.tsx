import React from "react";
import CacheFieldRenderer from "./CacheFieldRenderer";

interface CacheFieldGroupProps {
  title: string;
  fields: any[];
  cacheSettings: { [key: string]: any };
  redisType: string;
  gridCols?: string;
}

const CacheFieldGroup: React.FC<CacheFieldGroupProps> = ({
  title,
  fields,
  cacheSettings,
  redisType,
  gridCols = "grid-cols-1 gap-6 sm:grid-cols-2",
}) => {
  const shouldShowField = (field: any): boolean => {
    // Show field if it applies to all types (redis_type is null/undefined) or to current selected type
    if (field.redis_type === null || field.redis_type === undefined) {
      return true;
    }
    
    return field.redis_type === redisType;
  };

  const visibleFields = fields.filter(shouldShowField);

  if (visibleFields.length === 0) {
    return null;
  }

  return (
    <div className="space-y-6 pt-4 border-t border-gray-200">
      <h4 className="text-sm font-medium text-gray-900">{title}</h4>
      <div className={`grid ${gridCols}`}>
        {visibleFields.map((field) => {
          const currentValue = cacheSettings[field.field_name] ?? field.field_default ?? "";
          return (
            <CacheFieldRenderer
              key={field.field_name}
              field={field}
              currentValue={currentValue}
            />
          );
        })}
      </div>
    </div>
  );
};

export default CacheFieldGroup;

