import React from "react";
import LoggingSettings from "./LoggingSettings";

interface EditLoggingSettingsProps {
  value: any[];
  onChange: (value: any[]) => void;
  disabledCallbacks?: string[];
  onDisabledCallbacksChange?: (disabledCallbacks: string[]) => void;
}

/**
 * Wrapper component around LoggingSettings used for editing
 * a team's logging integrations.
 */
const EditLoggingSettings: React.FC<EditLoggingSettingsProps> = ({
  value,
  onChange,
  disabledCallbacks = [],
  onDisabledCallbacksChange,
}) => {
  return (
    <LoggingSettings
      value={value}
      onChange={onChange}
      disabledCallbacks={disabledCallbacks}
      onDisabledCallbacksChange={onDisabledCallbacksChange}
    />
  );
};

export default EditLoggingSettings;
