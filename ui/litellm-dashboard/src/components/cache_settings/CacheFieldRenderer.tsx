import React from "react";
import { TextInput, NumberInput } from "@tremor/react";

interface CacheFieldRendererProps {
  field: any;
  currentValue: any;
}

const CacheFieldRenderer: React.FC<CacheFieldRendererProps> = ({
  field,
  currentValue,
}) => {
  if (field.field_type === "Boolean") {
    return (
      <div className="space-y-2">
        <label className="text-sm font-medium text-gray-700">
          {field.ui_field_name}
        </label>
        <div className="flex items-center">
          <input
            type="checkbox"
            name={field.field_name}
            defaultChecked={currentValue === true || currentValue === "true"}
            className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
          />
          <span className="ml-2 text-sm text-gray-500">{field.field_description}</span>
        </div>
      </div>
    );
  }

  if (field.field_type === "List") {
    return (
      <div className="space-y-2">
        <label className="text-sm font-medium text-gray-700">
          {field.ui_field_name}
        </label>
        <textarea
          name={field.field_name}
          defaultValue={typeof currentValue === "object" ? JSON.stringify(currentValue, null, 2) : currentValue}
          placeholder={field.field_description}
          className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
          rows={4}
        />
        <p className="text-xs text-gray-500">{field.field_description}</p>
      </div>
    );
  }

  // Render number input for numeric fields
  if (field.field_type === "Integer" || field.field_type === "Float") {
    return (
      <div className="space-y-2">
        <label className="text-sm font-medium text-gray-700">
          {field.ui_field_name}
        </label>
        <NumberInput
          name={field.field_name}
          defaultValue={currentValue}
          placeholder={field.field_description}
          step={field.field_type === "Float" ? 0.01 : 1}
        />
        {field.field_description && (
          <p className="text-xs text-gray-500">{field.field_description}</p>
        )}
      </div>
    );
  }

  // Determine input type for text-based fields
  const inputType: "text" | "password" | "email" | "url" | undefined = 
    (field.field_name === "password" || field.field_name.includes("password")) 
      ? "password" 
      : "text";

  return (
    <div className="space-y-2">
      <label className="text-sm font-medium text-gray-700">
        {field.ui_field_name}
      </label>
      <TextInput
        name={field.field_name}
        type={inputType}
        defaultValue={currentValue}
        placeholder={field.field_description}
      />
      {field.field_description && (
        <p className="text-xs text-gray-500">{field.field_description}</p>
      )}
    </div>
  );
};

export default CacheFieldRenderer;

