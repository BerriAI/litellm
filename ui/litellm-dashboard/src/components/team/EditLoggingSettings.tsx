import React from 'react';
import LoggingSettings from './LoggingSettings';

interface EditLoggingSettingsProps {
  value: any[];
  onChange: (value: any[]) => void;
}

/**
 * Wrapper component around LoggingSettings used for editing
 * a team's logging integrations.
 */
const EditLoggingSettings: React.FC<EditLoggingSettingsProps> = ({ value, onChange }) => {
  return <LoggingSettings value={value} onChange={onChange} />;
};

export default EditLoggingSettings;
