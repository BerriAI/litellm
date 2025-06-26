import React from "react";
import { DateRangePicker, DateRangePickerValue, Text } from "@tremor/react";

interface UsageDatePickerProps {
  value: DateRangePickerValue;
  onValueChange: (value: DateRangePickerValue) => void;
  label?: string;
  className?: string;
  showTimeRange?: boolean;
}

/**
 * Reusable date picker component for usage dashboards.
 * Handles proper time boundaries for date ranges, especially "Today" selections.
 * Addresses timezone issues by ensuring UTC time boundaries are set correctly.
 */
const UsageDatePicker: React.FC<UsageDatePickerProps> = ({
  value,
  onValueChange,
  label = "Select Time Range",
  className = "",
  showTimeRange = true
}) => {
  const handleDateChange = (newValue: DateRangePickerValue) => {
    // Handle the case where "Today" or same-day selection is made
    if (newValue.from) {
      const adjustedValue = { ...newValue };
      
      // Create new Date objects to avoid mutating the original dates
      const adjustedStartTime = new Date(newValue.from);
      let adjustedEndTime: Date;
      
      if (newValue.to) {
        adjustedEndTime = new Date(newValue.to);
      } else {
        // If no end date is provided (like "Today" from dropdown), use the same date
        adjustedEndTime = new Date(newValue.from);
      }
      
      // Check if it's the same day (Today selection or single day selection)
      const isSameDay = 
        adjustedStartTime.toDateString() === adjustedEndTime.toDateString();
      
      if (isSameDay) {
        // For same-day selections, set proper time boundaries
        // Use local timezone boundaries that will be converted to UTC properly
        adjustedStartTime.setHours(0, 0, 0, 0); // Start of day in local time
        adjustedEndTime.setHours(23, 59, 59, 999); // End of day in local time
      } else {
        // For multi-day ranges, set start to beginning of first day and end to end of last day
        adjustedStartTime.setHours(0, 0, 0, 0);
        adjustedEndTime.setHours(23, 59, 59, 999);
      }
      
      adjustedValue.from = adjustedStartTime;
      adjustedValue.to = adjustedEndTime;
      
      onValueChange(adjustedValue);
    } else {
      // If no from date, pass through as-is
      onValueChange(newValue);
    }
  };

  const formatTimeRange = (from: Date | undefined, to: Date | undefined) => {
    if (!from || !to) return "";
    
    const formatDateTime = (date: Date) => {
      return date.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        hour12: true,
        timeZoneName: 'short'
      });
    };

    const isSameDay = from.toDateString() === to.toDateString();
    
    if (isSameDay) {
      const dateStr = from.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric'
      });
      const startTime = from.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        hour12: true
      });
      const endTime = to.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        hour12: true,
        timeZoneName: 'short'
      });
      return `${dateStr}: ${startTime} - ${endTime}`;
    } else {
      return `${formatDateTime(from)} - ${formatDateTime(to)}`;
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
      {showTimeRange && value.from && value.to && (
        <Text className="mt-1 text-xs text-gray-500">
          {formatTimeRange(value.from, value.to)}
        </Text>
      )}
    </div>
  );
};

export default UsageDatePicker;