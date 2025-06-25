import React from "react";
import { DateRangePicker, DateRangePickerValue, Text } from "@tremor/react";

interface UsageDatePickerProps {
  value: DateRangePickerValue;
  onValueChange: (value: DateRangePickerValue) => void;
  label?: string;
  className?: string;
}

/**
 * Reusable date picker component for usage dashboards.
 * Handles proper time boundaries for date ranges, especially "Today" selections.
 */
const UsageDatePicker: React.FC<UsageDatePickerProps> = ({
  value,
  onValueChange,
  label = "Select Time Range",
  className = ""
}) => {
  const handleDateChange = (newValue: DateRangePickerValue) => {
    // Handle the case where "Today" or same-day selection is made
    if (newValue.from && newValue.to) {
      const adjustedValue = { ...newValue };
      
      // Create new Date objects to avoid mutating the original dates
      const adjustedStartTime = new Date(newValue.from);
      const adjustedEndTime = new Date(newValue.to);
      
      // Check if it's the same day (Today selection)
      const isSameDay = 
        adjustedStartTime.toDateString() === adjustedEndTime.toDateString();
      
      if (isSameDay) {
        // For same-day selections (like "Today"), set proper time boundaries
        adjustedStartTime.setHours(0, 0, 0, 0); // Start of day
        adjustedEndTime.setHours(23, 59, 59, 999); // End of day
      } else {
        // For multi-day ranges, set start to beginning of first day and end to end of last day
        adjustedStartTime.setHours(0, 0, 0, 0);
        adjustedEndTime.setHours(23, 59, 59, 999);
      }
      
      adjustedValue.from = adjustedStartTime;
      adjustedValue.to = adjustedEndTime;
      
      onValueChange(adjustedValue);
    } else {
      // If either date is missing, pass through as-is
      onValueChange(newValue);
    }
  };

  return (
    <div className={className}>
      {label && <Text className="mb-2">{label}</Text>}
      <DateRangePicker
        enableSelect={true}
        value={value}
        onValueChange={handleDateChange}
      />
    </div>
  );
};

export default UsageDatePicker;