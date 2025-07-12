import React from 'react';
import LoggingSettings from './LoggingSettings';

interface EditLoggingSettingsProps {
  value: any[];
  onChange: (value: any[]) => void;
  disabledCallbacks?: string[];
  onDisabledCallbacksChange?: (disabled: string[]) => void;
  accessToken?: string;
}

/**
 * Wrapper component around LoggingSettings used for editing
 * a team's logging integrations and disabled callbacks.
 */
const EditLoggingSettings: React.FC<EditLoggingSettingsProps> = ({ 
  value, 
  onChange, 
  disabledCallbacks,
  onDisabledCallbacksChange,
  accessToken
}) => {
  return (
    <LoggingSettings 
      value={value} 
      onChange={onChange}
      disabledCallbacks={disabledCallbacks}
      onDisabledCallbacksChange={onDisabledCallbacksChange}
      accessToken={accessToken}
    />
  );
};

export default EditLoggingSettings;
