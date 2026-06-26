import { CalendarOutlined, ClockCircleOutlined } from "@ant-design/icons";
import { Button, DateRangePickerValue, Text } from "@tremor/react";
import moment from "moment";
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { TFunction } from "i18next";

interface AdvancedDatePickerProps {
  value: DateRangePickerValue;
  onValueChange: (value: DateRangePickerValue) => void;
  label?: string;
  className?: string;
  showTimeRange?: boolean;
}

interface RelativeTimeOption {
  label: string;
  shortLabel: string;
  getValue: () => { from: Date; to: Date };
}

const getRelativeTimeOptions = (t: TFunction): RelativeTimeOption[] => [
  {
    label: t("shared.advancedDatePicker.today"),
    shortLabel: t("shared.advancedDatePicker.todayShort"),
    getValue: () => ({
      from: moment().startOf("day").toDate(),
      to: moment().endOf("day").toDate(),
    }),
  },
  {
    label: t("shared.advancedDatePicker.last7Days"),
    shortLabel: t("shared.advancedDatePicker.last7DaysShort"),
    getValue: () => ({
      from: moment().subtract(7, "days").startOf("day").toDate(),
      to: moment().endOf("day").toDate(),
    }),
  },
  {
    label: t("shared.advancedDatePicker.last30Days"),
    shortLabel: t("shared.advancedDatePicker.last30DaysShort"),
    getValue: () => ({
      from: moment().subtract(30, "days").startOf("day").toDate(),
      to: moment().endOf("day").toDate(),
    }),
  },
  {
    label: t("shared.advancedDatePicker.monthToDate"),
    shortLabel: t("shared.advancedDatePicker.monthToDateShort"),
    getValue: () => ({
      from: moment().startOf("month").toDate(),
      to: moment().endOf("day").toDate(),
    }),
  },
  {
    label: t("shared.advancedDatePicker.yearToDate"),
    shortLabel: t("shared.advancedDatePicker.yearToDateShort"),
    getValue: () => ({
      from: moment().startOf("year").toDate(),
      to: moment().endOf("day").toDate(),
    }),
  },
];

const AdvancedDatePicker: React.FC<AdvancedDatePickerProps> = ({
  value,
  onValueChange,
  label,
  showTimeRange = true,
}) => {
  const { t } = useTranslation();
  const relativeTimeOptions = useMemo(() => getRelativeTimeOptions(t), [t]);
  const resolvedLabel = label !== undefined ? label : t("shared.advancedDatePicker.selectTimeRange");

  const [isOpen, setIsOpen] = useState(false);
  const [tempValue, setTempValue] = useState<DateRangePickerValue>(value);
  const [selectedOption, setSelectedOption] = useState<string | null>(null);

  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");

  const dropdownRef = useRef<HTMLDivElement>(null);

  const getMatchingOption = useCallback(
    (currentValue: DateRangePickerValue): string | null => {
      if (!currentValue.from || !currentValue.to) return null;

      for (const option of relativeTimeOptions) {
        const optionRange = option.getValue();

        const fromMatches = moment(currentValue.from).isSame(moment(optionRange.from), "day");
        const toMatches = moment(currentValue.to).isSame(moment(optionRange.to), "day");

        if (fromMatches && toMatches) {
          return option.shortLabel;
        }
      }

      return null;
    },
    [relativeTimeOptions],
  );

  useEffect(() => {
    const matchingOption = getMatchingOption(value);
    setSelectedOption(matchingOption);
  }, [value, getMatchingOption]);

  const validateDateRange = useCallback(() => {
    if (!startDate || !endDate) {
      return { isValid: true, error: "" };
    }

    const start = moment(startDate, "YYYY-MM-DD");
    const end = moment(endDate, "YYYY-MM-DD");

    if (!start.isValid() || !end.isValid()) {
      return { isValid: false, error: t("shared.advancedDatePicker.invalidDateFormat") };
    }

    if (end.isBefore(start)) {
      return { isValid: false, error: t("shared.advancedDatePicker.endBeforeStart") };
    }

    return { isValid: true, error: "" };
  }, [startDate, endDate, t]);

  const validation = validateDateRange();

  useEffect(() => {
    if (value.from) {
      setStartDate(moment(value.from).format("YYYY-MM-DD"));
    }
    if (value.to) {
      setEndDate(moment(value.to).format("YYYY-MM-DD"));
    }
    setTempValue(value);
  }, [value]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isOpen]);

  const formatDisplayRange = useCallback(
    (from: Date | undefined, to: Date | undefined) => {
      if (!from || !to) return t("shared.advancedDatePicker.selectDateRange");

      const formatDateTime = (date: Date) => {
        return moment(date).format("D MMM, HH:mm");
      };

      return `${formatDateTime(from)} - ${formatDateTime(to)}`;
    },
    [t],
  );

  const adjustDateRange = useCallback((newValue: DateRangePickerValue): DateRangePickerValue => {
    if (!newValue.from) return newValue;

    const adjustedValue = { ...newValue };
    const adjustedStartTime = new Date(newValue.from);
    let adjustedEndTime: Date;

    if (newValue.to) {
      adjustedEndTime = new Date(newValue.to);
    } else {
      adjustedEndTime = new Date(newValue.from);
    }

    adjustedStartTime.setHours(0, 0, 0, 0);
    adjustedEndTime.setHours(23, 59, 59, 999);

    adjustedValue.from = adjustedStartTime;
    adjustedValue.to = adjustedEndTime;

    return adjustedValue;
  }, []);

  const handleRelativeTimeSelect = (option: RelativeTimeOption) => {
    const { from, to } = option.getValue();
    const newValue = { from, to };

    setTempValue(newValue);
    setSelectedOption(option.shortLabel);

    setStartDate(moment(from).format("YYYY-MM-DD"));
    setEndDate(moment(to).format("YYYY-MM-DD"));
  };

  const updateTempValueFromInputs = useCallback(() => {
    try {
      if (startDate && endDate && validation.isValid) {
        const from = moment(startDate, "YYYY-MM-DD").startOf("day");
        const to = moment(endDate, "YYYY-MM-DD").endOf("day");

        if (from.isValid() && to.isValid()) {
          const newValue = { from: from.toDate(), to: to.toDate() };
          setTempValue(newValue);

          const matchingOption = getMatchingOption(newValue);
          setSelectedOption(matchingOption);
        }
      }
    } catch (error) {
      console.warn("Invalid date format:", error);
    }
  }, [startDate, endDate, validation.isValid, getMatchingOption]);

  useEffect(() => {
    updateTempValueFromInputs();
  }, [updateTempValueFromInputs]);

  const handleApply = () => {
    if (tempValue.from && tempValue.to && validation.isValid) {
      onValueChange(tempValue);

      requestIdleCallback(
        () => {
          const adjustedValue = adjustDateRange(tempValue);
          onValueChange(adjustedValue);
        },
        { timeout: 100 },
      );

      setIsOpen(false);
    }
  };

  const handleCancel = () => {
    setTempValue(value);

    if (value.from) {
      setStartDate(moment(value.from).format("YYYY-MM-DD"));
    }
    if (value.to) {
      setEndDate(moment(value.to).format("YYYY-MM-DD"));
    }

    const matchingOption = getMatchingOption(value);
    setSelectedOption(matchingOption);

    setIsOpen(false);
  };

  return (
    <div className="flex items-center gap-3">
      {resolvedLabel && <Text className="text-sm font-medium text-gray-700 whitespace-nowrap">{resolvedLabel}</Text>}
      <div className="relative" ref={dropdownRef}>
        <div
          className="w-[300px] px-3 py-2 text-sm border border-gray-300 rounded-md bg-white cursor-pointer hover:border-gray-400 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
          onClick={() => setIsOpen(!isOpen)}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <ClockCircleOutlined className="text-gray-600" />
              <span className="text-gray-900">{formatDisplayRange(value.from, value.to)}</span>
            </div>
            <svg
              className={`w-4 h-4 text-gray-400 transition-transform ${isOpen ? "rotate-180" : ""}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </div>
        </div>

        {isOpen && (
          <div className="absolute top-full right-0 z-[9999] min-w-[600px] mt-1 bg-white border border-gray-200 rounded-lg shadow-xl">
            <div className="flex">
              <div className="w-1/2 border-r border-gray-200">
                <div className="p-3 border-b border-gray-200">
                  <span className="text-sm font-semibold text-gray-900">
                    {t("shared.advancedDatePicker.relativeTime")}
                  </span>
                </div>
                <div className="h-[350px] overflow-y-auto">
                  {relativeTimeOptions.map((option) => {
                    const isSelected = selectedOption === option.shortLabel;
                    return (
                      <div
                        key={option.shortLabel}
                        className={`flex items-center justify-between px-5 py-4 cursor-pointer border-b border-gray-100 transition-colors ${
                          isSelected ? "bg-blue-50 hover:bg-blue-100 border-blue-200" : "hover:bg-gray-50"
                        }`}
                        onClick={() => handleRelativeTimeSelect(option)}
                      >
                        <span className={`text-sm ${isSelected ? "text-blue-700 font-medium" : "text-gray-700"}`}>
                          {option.label}
                        </span>
                        <span
                          className={`text-xs px-2 py-1 rounded capitalize ${
                            isSelected ? "text-blue-700 bg-blue-100" : "text-gray-500 bg-gray-100"
                          }`}
                        >
                          {option.shortLabel}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="w-1/2 relative">
                <div className="p-3.5 border-b border-gray-200">
                  <div className="flex items-center gap-2">
                    <CalendarOutlined className="text-gray-600" />
                    <span className="text-sm font-semibold text-gray-900">
                      {t("shared.advancedDatePicker.startAndEndDates")}
                    </span>
                  </div>
                </div>

                <div className="p-6 space-y-6 pb-20">
                  <div>
                    <label className="text-sm text-gray-700 mb-1 block">
                      {t("shared.advancedDatePicker.startDate")}
                    </label>
                    <input
                      type="date"
                      value={startDate}
                      onChange={(e) => setStartDate(e.target.value)}
                      className={`w-65 px-3 py-2 text-sm border rounded-md cursor-pointer hover:border-gray-400 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 ${
                        !validation.isValid
                          ? "border-red-300 focus:border-red-500 focus:ring-red-200"
                          : "border-gray-300"
                      }`}
                    />
                  </div>

                  <div>
                    <label className="text-sm text-gray-700 mb-1 block">{t("shared.advancedDatePicker.endDate")}</label>
                    <input
                      type="date"
                      value={endDate}
                      onChange={(e) => setEndDate(e.target.value)}
                      className={`w-65 px-3 py-2 text-sm border rounded-md cursor-pointer hover:border-gray-400 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 ${
                        !validation.isValid
                          ? "border-red-300 focus:border-red-500 focus:ring-red-200"
                          : "border-gray-300"
                      }`}
                    />
                  </div>

                  {!validation.isValid && validation.error && (
                    <div className="bg-red-50 border border-red-200 rounded-md p-3">
                      <div className="flex items-center gap-2">
                        <svg className="w-4 h-4 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z"
                          />
                        </svg>
                        <span className="text-sm text-red-700 font-medium">{validation.error}</span>
                      </div>
                    </div>
                  )}

                  {tempValue.from && tempValue.to && validation.isValid && (
                    <div className="bg-blue-50 p-3 rounded-md space-y-1">
                      <div className="text-xs text-blue-800">
                        <span className="font-medium">{t("shared.advancedDatePicker.from")}</span>{" "}
                        {moment(tempValue.from).format("MMM D, YYYY [at] HH:mm:ss")}
                      </div>
                      <div className="text-xs text-blue-800">
                        <span className="font-medium">{t("shared.advancedDatePicker.to")}</span>{" "}
                        {moment(tempValue.to).format("MMM D, YYYY [at] HH:mm:ss")}
                      </div>
                    </div>
                  )}
                </div>

                <div className="absolute bottom-4 right-4">
                  <div className="flex gap-2">
                    <Button variant="secondary" onClick={handleCancel}>
                      {t("common.cancel")}
                    </Button>
                    <Button onClick={handleApply} disabled={!tempValue.from || !tempValue.to || !validation.isValid}>
                      {t("common.apply")}
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AdvancedDatePicker;
