import React, { useCallback, useState, useRef } from "react";
import { DateRangePicker, DateRangePickerValue, Text } from "@tremor/react";

interface UsageDatePickerProps {
  value: DateRangePickerValue;
  onValueChange: (value: DateRangePickerValue) => void;
  label?: string;
  className?: string;
  showTimeRange?: boolean;
}

/**
 * Ultra responsive date picker with instant click feedback
 */
const UsageDatePicker: React.FC<UsageDatePickerProps> = ({
  value,
  onValueChange,
  label = "Select Time Range",
  className = "",
  showTimeRange = true,
}) => {
  const [showSelectedFeedback, setShowSelectedFeedback] = useState(false);
  const datePickerRef = useRef<HTMLDivElement>(null);

  // This only triggers AFTER user has actually made a selection
  const handleDateChange = useCallback(
    (newValue: DateRangePickerValue) => {
      // Show "Selected" feedback ONLY after actual selection is made
      setShowSelectedFeedback(true);

      // Hide the feedback after a short time
      setTimeout(() => setShowSelectedFeedback(false), 1500);

      // Update parent immediately
      onValueChange(newValue);

      // Do heavy processing in background
      requestIdleCallback(
        () => {
          if (newValue.from) {
            const adjustedValue = { ...newValue };
            const adjustedStartTime = new Date(newValue.from);
            let adjustedEndTime: Date;

            if (newValue.to) {
              adjustedEndTime = new Date(newValue.to);
            } else {
              adjustedEndTime = new Date(newValue.from);
            }

            const isSameDay = adjustedStartTime.toDateString() === adjustedEndTime.toDateString();

            if (isSameDay) {
              adjustedStartTime.setHours(0, 0, 0, 0);
              adjustedEndTime.setHours(23, 59, 59, 999);
            } else {
              adjustedStartTime.setHours(0, 0, 0, 0);
              adjustedEndTime.setHours(23, 59, 59, 999);
            }

            adjustedValue.from = adjustedStartTime;
            adjustedValue.to = adjustedEndTime;
            onValueChange(adjustedValue);
          }
        },
        { timeout: 100 },
      );
    },
    [onValueChange],
  );

  const formatTimeRange = useCallback((from: Date | undefined, to: Date | undefined) => {
    if (!from || !to) return "";

    const formatDateTime = (date: Date) => {
      return date.toLocaleString("en-US", {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
        hour12: true,
        timeZoneName: "short",
      });
    };

    const isSameDay = from.toDateString() === to.toDateString();

    if (isSameDay) {
      const dateStr = from.toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
      });
      const startTime = from.toLocaleTimeString("en-US", {
        hour: "2-digit",
        minute: "2-digit",
        hour12: true,
      });
      const endTime = to.toLocaleTimeString("en-US", {
        hour: "2-digit",
        minute: "2-digit",
        hour12: true,
        timeZoneName: "short",
      });
      return `${dateStr}: ${startTime} - ${endTime}`;
    } else {
      return `${formatDateTime(from)} - ${formatDateTime(to)}`;
    }
  }, []);

  return (
    <div className={className}>
      {label && <Text className="mb-2">{label}</Text>}

      {/* Container with relative positioning for absolute placement */}
      <div className="relative w-fit">
        <div ref={datePickerRef}>
          <DateRangePicker
            enableSelect={true}
            value={value}
            onValueChange={handleDateChange} // Only triggers on actual selection
            placeholder="Select date range"
            enableClear={false}
            style={{ zIndex: 100 }}
          />
        </div>

        {/* ONLY SHOW AFTER ACTUAL SELECTION IS COMPLETED */}
        {showSelectedFeedback && (
          <div
            className="absolute top-1/2 animate-pulse"
            style={{
              left: "calc(100% + 8px)",
              transform: "translateY(-50%)",
              zIndex: 110,
            }}
          >
            <div className="flex items-center gap-1 text-green-600 text-sm font-medium bg-white px-2 py-1 rounded-full border border-green-200 shadow-sm whitespace-nowrap">
              <div className="w-3 h-3 bg-green-500 text-white rounded-full flex items-center justify-center text-xs">
                ✓
              </div>
              <span className="text-xs">Selected</span>
            </div>
          </div>
        )}
      </div>

      {/* Time range display */}
      {showTimeRange && value.from && value.to && (
        <Text className="mt-2 text-xs text-gray-500">{formatTimeRange(value.from, value.to)}</Text>
      )}
    </div>
  );
};

export default UsageDatePicker;
