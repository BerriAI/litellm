import moment from "moment";
import { useEffect, useRef, useState } from "react";
import { SyncOutlined } from "@ant-design/icons";
import { Button, Switch } from "antd";
import { QUICK_SELECT_OPTIONS } from "./constants";
import { getTimeRangeDisplay } from "./logs_utils";
import type { PaginatedResponse } from "./log_filter_logic";

interface LogsTableToolbarProps {
  searchTerm: string;
  onSearchChange: (value: string) => void;
  startTime: string;
  onStartTimeChange: (value: string) => void;
  endTime: string;
  onEndTimeChange: (value: string) => void;
  isCustomDate: boolean;
  onIsCustomDateChange: (value: boolean) => void;
  selectedTimeInterval: { value: number; unit: string };
  onSelectedTimeIntervalChange: (value: { value: number; unit: string }) => void;
  isLiveTail: boolean;
  onIsLiveTailChange: (value: boolean) => void;
  currentPage: number;
  onCurrentPageChange: (updater: number | ((prev: number) => number)) => void;
  pageSize: number;
  isLoading: boolean;
  isButtonLoading: boolean;
  onRefetch: () => void;
  filteredLogs: PaginatedResponse;
}

export function LogsTableToolbar({
  searchTerm,
  onSearchChange,
  startTime,
  onStartTimeChange,
  endTime,
  onEndTimeChange,
  isCustomDate,
  onIsCustomDateChange,
  selectedTimeInterval,
  onSelectedTimeIntervalChange,
  isLiveTail,
  onIsLiveTailChange,
  currentPage,
  onCurrentPageChange,
  pageSize,
  isLoading,
  isButtonLoading,
  onRefetch,
  filteredLogs,
}: LogsTableToolbarProps) {
  const [quickSelectOpen, setQuickSelectOpen] = useState(false);
  const quickSelectRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (quickSelectRef.current && !quickSelectRef.current.contains(event.target as Node)) {
        setQuickSelectOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const selectedOption = QUICK_SELECT_OPTIONS.find(
    (option) => option.value === selectedTimeInterval.value && option.unit === selectedTimeInterval.unit,
  );
  const displayLabel = isCustomDate ? getTimeRangeDisplay(isCustomDate, startTime, endTime) : selectedOption?.label;

  return (
    <>
      <div className="border-b px-6 py-4 w-full max-w-full box-border">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between space-y-4 md:space-y-0 w-full max-w-full box-border">
          <div className="flex flex-wrap items-center gap-3 w-full max-w-full box-border">
            <div className="relative w-64 min-w-0 flex-shrink-0">
              <input
                type="text"
                placeholder="Search by Request ID"
                className="w-full px-3 py-2 pl-8 border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                value={searchTerm}
                onChange={(e) => onSearchChange(e.target.value)}
              />
              <svg
                className="absolute left-2.5 top-2.5 h-4 w-4 text-gray-500"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                />
              </svg>
            </div>

            <div className="flex items-center gap-2 min-w-0 flex-shrink">
              <div className="relative z-50" ref={quickSelectRef}>
                <button
                  onClick={() => setQuickSelectOpen(!quickSelectOpen)}
                  className="px-3 py-2 text-sm border rounded-md hover:bg-gray-50 flex items-center gap-2"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
                    />
                  </svg>
                  {displayLabel}
                </button>

                {quickSelectOpen && (
                  <div className="absolute left-0 mt-2 w-64 bg-white rounded-lg shadow-lg border p-2 z-50">
                    <div className="space-y-1">
                      {QUICK_SELECT_OPTIONS.map((option) => (
                        <button
                          key={option.label}
                          className={`w-full px-3 py-2 text-left text-sm hover:bg-gray-50 rounded-md ${displayLabel === option.label ? "bg-blue-50 text-blue-600" : ""}`}
                          onClick={() => {
                            onCurrentPageChange(1);
                            onEndTimeChange(moment().format("YYYY-MM-DDTHH:mm"));
                            onStartTimeChange(
                              moment()
                                .subtract(option.value, option.unit as any)
                                .format("YYYY-MM-DDTHH:mm"),
                            );
                            onSelectedTimeIntervalChange({ value: option.value, unit: option.unit });
                            onIsCustomDateChange(false);
                            setQuickSelectOpen(false);
                          }}
                        >
                          {option.label}
                        </button>
                      ))}
                      <div className="border-t my-2" />
                      <button
                        className={`w-full px-3 py-2 text-left text-sm hover:bg-gray-50 rounded-md ${isCustomDate ? "bg-blue-50 text-blue-600" : ""}`}
                        onClick={() => onIsCustomDateChange(!isCustomDate)}
                      >
                        Custom Range
                      </button>
                    </div>
                  </div>
                )}
              </div>

              <div className="flex items-center gap-2">
                <span className="text-sm font-medium text-gray-900">Live Tail</span>
                <Switch checked={isLiveTail} defaultChecked={true} onChange={onIsLiveTailChange} />
              </div>

              <Button
                type="default"
                icon={<SyncOutlined spin={isButtonLoading} />}
                onClick={onRefetch}
                disabled={isButtonLoading}
                title="Fetch data"
              >
                {isButtonLoading ? "Fetching" : "Fetch"}
              </Button>
            </div>

            {isCustomDate && (
              <div className="flex items-center gap-2">
                <div>
                  <input
                    type="datetime-local"
                    value={startTime}
                    onChange={(e) => {
                      onStartTimeChange(e.target.value);
                      onCurrentPageChange(1);
                    }}
                    className="px-3 py-2 border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
                <span className="text-gray-500">to</span>
                <div>
                  <input
                    type="datetime-local"
                    value={endTime}
                    onChange={(e) => {
                      onEndTimeChange(e.target.value);
                      onCurrentPageChange(1);
                    }}
                    className="px-3 py-2 border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
              </div>
            )}
          </div>

          <div className="flex items-center space-x-4">
            <span className="text-sm text-gray-700 whitespace-nowrap">
              Showing {isLoading ? "..." : filteredLogs ? (currentPage - 1) * pageSize + 1 : 0} -{" "}
              {isLoading
                ? "..."
                : filteredLogs
                  ? Math.min(currentPage * pageSize, filteredLogs.total)
                  : 0}{" "}
              of {isLoading ? "..." : filteredLogs ? filteredLogs.total : 0} results
            </span>
            <div className="flex items-center space-x-2">
              <span className="text-sm text-gray-700 min-w-[90px]">
                Page {isLoading ? "..." : currentPage} of{" "}
                {isLoading ? "..." : filteredLogs ? filteredLogs.total_pages : 1}
              </span>
              <button
                onClick={() => onCurrentPageChange((p: number) => Math.max(1, p - 1))}
                disabled={isLoading || currentPage === 1}
                className="px-3 py-1 text-sm border rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Previous
              </button>
              <button
                onClick={() => onCurrentPageChange((p: number) => Math.min(filteredLogs.total_pages || 1, p + 1))}
                disabled={isLoading || currentPage === (filteredLogs.total_pages || 1)}
                className="px-3 py-1 text-sm border rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Next
              </button>
            </div>
          </div>
        </div>
      </div>
      {isLiveTail && currentPage === 1 && (
        <div className="mb-4 px-4 py-2 bg-green-50 border border-green-200 rounded-md flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-sm text-green-700">Auto-refreshing every 15 seconds</span>
          </div>
          <button
            onClick={() => onIsLiveTailChange(false)}
            className="text-sm text-green-600 hover:text-green-800"
          >
            Stop
          </button>
        </div>
      )}
    </>
  );
}
