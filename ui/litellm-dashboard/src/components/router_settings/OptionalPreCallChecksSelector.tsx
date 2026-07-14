import React from "react";
import { Select } from "antd";

interface OptionalPreCallChecksSelectorProps {
  value: string[];
  options: string[];
  routerFieldsMetadata: { [key: string]: any };
  onChange: (value: string[]) => void;
}

const OptionalPreCallChecksSelector: React.FC<OptionalPreCallChecksSelectorProps> = ({
  value,
  options,
  routerFieldsMetadata,
  onChange,
}) => {
  const meta = routerFieldsMetadata["optional_pre_call_checks"];

  return (
    <div className="space-y-2 max-w-3xl">
      <label className="block">
        <span className="text-xs font-medium text-gray-700 uppercase tracking-wide">
          {meta?.ui_field_name || "Optional Pre-call Checks"}
        </span>
        <p className="text-xs text-gray-500 mt-0.5 mb-2">
          {meta?.field_description || ""}
          {meta?.link && (
            <>
              {" "}
              <a
                href={meta.link}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:text-blue-800 underline"
              >
                Learn more
              </a>
            </>
          )}
        </p>
        <Select
          mode="multiple"
          value={value}
          onChange={onChange}
          options={options.map((option) => ({ value: option, label: option }))}
          placeholder="No pre-call checks enabled"
          className="w-full"
          data-testid="optional-pre-call-checks-select"
        />
      </label>
    </div>
  );
};

export default OptionalPreCallChecksSelector;
