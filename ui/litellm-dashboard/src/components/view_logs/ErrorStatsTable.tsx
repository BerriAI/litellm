"use client";

import React, { useState, useMemo, useCallback } from "react";
import { Card, Title, BarList, Text } from "@tremor/react";
import Chart from "react-apexcharts";

/**
 * Represents a single error statistic data point from the backend.
 * Each row contains the time bucket, error classification, and occurrence count.
 */
interface ErrorStat {
  time_bucket: string;
  extracted_error: string;
  count: number;
}

/**
 * Props for the ErrorStatsTable component.
 * @param data - Array of error statistics aggregated by time bucket
 * @param timeBucketSize - Optional string describing the bucket size (e.g., "1 hour", "1 day")
 * @param onTimeRangeSelect - Callback when user selects a time range on the chart
 * @param setCurrentPage - Callback to reset page to 1 when time range changes
 * @param setIsCustomDate - Callback to set custom date mode when user selects a range
 * @param onSelectedCategoriesChange - Callback when selected error categories change
 */
interface ErrorStatsTableProps {
  data: ErrorStat[];
  timeBucketSize?: string;
  onTimeRangeSelect?: (startTime: string, endTime: string) => void;
  setCurrentPage?: (page: number) => void;
  setIsCustomDate?: (isCustom: boolean) => void;
  onSelectedCategoriesChange?: (categories: string[]) => void;
}

/**
 * Color palette for error category visualization.
 * Colors are assigned cyclically to error classes.
 * Using hex codes compatible with ApexCharts.
 */
const COLORS = [
  "#EF4444", // red
  "#F97316", // orange
  "#F59E0B", // amber
  "#EAB308", // yellow
  "#84CC16", // lime
  "#22C55E", // green
  "#10B981", // emerald
  "#14B8A6", // teal
  "#06B6D4", // cyan
  "#0EA5E9", // sky
  "#3B82F6", // blue
  "#6366F1", // indigo
  "#8B5CF6", // violet
  "#A855F7", // purple
  "#D946EF", // fuchsia
  "#EC4899", // pink
  "#F43F5E", // rose
  "#64748B", // slate
  "#6B7280", // gray
  "#71717A", // zinc
  "#737373", // neutral
  "#78716C", // stone
];

/**
 * Tremor color names corresponding to the hex colors.
 * BarList requires Tremor color names, not hex codes.
 */
const TREMOR_COLORS = [
  "red",
  "orange",
  "amber",
  "yellow",
  "lime",
  "green",
  "emerald",
  "teal",
  "cyan",
  "sky",
  "blue",
  "indigo",
  "violet",
  "purple",
  "fuchsia",
  "pink",
  "rose",
  "slate",
  "gray",
  "zinc",
  "neutral",
  "stone",
];

/**
 * Mapping of time bucket display strings to millisecond values.
 * Matches the bucket intervals defined in spend_management_endpoints.py.
 * The backend uses these to dynamically scale buckets based on the time range.
 */
const BUCKET_INTERVALS_MS: Record<string, number> = {
  "1 minute": 60 * 1000,
  "2 minutes": 2 * 60 * 1000,
  "5 minutes": 5 * 60 * 1000,
  "10 minutes": 10 * 60 * 1000,
  "15 minutes": 15 * 60 * 1000,
  "30 minutes": 30 * 60 * 1000,
  "1 hour": 60 * 60 * 1000,
  "2 hours": 2 * 60 * 60 * 1000,
  "4 hours": 4 * 60 * 60 * 1000,
  "8 hours": 8 * 60 * 60 * 1000,
  "12 hours": 12 * 60 * 60 * 1000,
  "1 day": 24 * 60 * 60 * 1000,
  "2 days": 2 * 24 * 60 * 60 * 1000,
  "3 days": 3 * 24 * 60 * 60 * 1000,
  "1 week": 7 * 24 * 60 * 60 * 1000,
  "2 weeks": 14 * 24 * 60 * 60 * 1000,
  "1 month": 30 * 24 * 60 * 60 * 1000, // Approximate: 30 days
};

export const ErrorStatsTable: React.FC<ErrorStatsTableProps> = ({ data, timeBucketSize, onTimeRangeSelect, setCurrentPage, setIsCustomDate, onSelectedCategoriesChange }) => {
  // State to track which error categories are selected for filtering/highlighting
  const [selectedCategories, setSelectedCategories] = useState<string[]>([]);

  // Notify parent when selected categories change
  React.useEffect(() => {
    onSelectedCategoriesChange?.(selectedCategories);
  }, [selectedCategories, onSelectedCategoriesChange]);

  /**
   * Extracts unique error categories from the data.
   * Each distinct error class becomes a category in the chart.
   */
  const errorCategories = useMemo(() => {
    const uniqueErrors = new Set(data.map(item => item.extracted_error || "Unknown Error"));
    return Array.from(uniqueErrors);
  }, [data]);

  /**
   * Assigns a consistent color to each error category.
   * Colors are assigned based on the order of categories to maintain consistency.
   * Returns both hex (for ApexCharts) and Tremor color name (for BarList).
   */
  const categoryColors = useMemo(() => {
    const mapping: Record<string, { hex: string; tremor: string }> = {};
    errorCategories.forEach((cat, index) => {
      mapping[cat] = {
        hex: COLORS[index % COLORS.length],
        tremor: TREMOR_COLORS[index % TREMOR_COLORS.length],
      };
    });
    return mapping;
  }, [errorCategories]);

  /**
   * Aggregates error counts by category for the bar list display.
   * Categories with no selections are grayed out.
   */
  const errorStats = useMemo(() => {
    const totals: Record<string, number> = {};
    data.forEach(item => {
      const errorClass = item.extracted_error || "Unknown Error";
      totals[errorClass] = (totals[errorClass] || 0) + item.count;
    });
    return Object.entries(totals)
      .map(([name, value]) => ({
        name,
        value,
        color: selectedCategories.length === 0 || selectedCategories.includes(name)
          ? categoryColors[name]?.tremor || "gray"
          : "gray"
      }))
      .sort((a, b) => b.value - a.value); // Sort by descending count
  }, [data, selectedCategories, categoryColors]);

  /**
   * Prepares time-series data for the area chart with uniform x-axis spacing.
   *
   * Key steps:
   * 1. Parse and aggregate data by timestamp
   * 2. Determine the bucket interval (from props or infer from data)
   * 3. Fill in missing time buckets with zero values to ensure uniform spacing
   * 4. Format dates for display
   */
  const chartData = useMemo(() => {
    if (data.length === 0) return [];

    // Step 1: Aggregate data by timestamp (milliseconds)
    const timeMap = new Map<number, Record<string, number>>();

    data.forEach((item) => {
      if (!item.time_bucket) return;
      try {
        const date = new Date(item.time_bucket);
        const timestamp = date.getTime();

        if (!timeMap.has(timestamp)) {
          timeMap.set(timestamp, {});
        }

        const errorClass = item.extracted_error || "Unknown Error";
        const bucket = timeMap.get(timestamp)!;
        bucket[errorClass] = (bucket[errorClass] || 0) + item.count;
      } catch (e) {
        console.error("Error parsing date:", e);
      }
    });

    // Step 2: Determine bucket interval in milliseconds
    // Priority: 1) From timeBucketSize prop, 2) Infer from data, 3) Default to 1 hour
    const bucketMs = (() => {
      // Try to parse from the timeBucketSize prop (e.g., "1 hour", "2 days", "1 month")
      if (timeBucketSize) {
        // First, try direct lookup in our predefined intervals
        if (BUCKET_INTERVALS_MS[timeBucketSize]) {
          return BUCKET_INTERVALS_MS[timeBucketSize];
        }
        // Fallback: parse pattern like "X minutes/hours/days/weeks/months"
        const match = timeBucketSize.match(/^(\d+)\s*(minute|hour|day|week|month)s?$/i);
        if (match) {
          const value = parseInt(match[1], 10);
          const unit = match[2].toLowerCase();
          const multipliers: Record<string, number> = {
            minute: 60 * 1000,
            hour: 60 * 60 * 1000,
            day: 24 * 60 * 60 * 1000,
            week: 7 * 24 * 60 * 60 * 1000,
            month: 30 * 24 * 60 * 60 * 1000,
          };
          return value * multipliers[unit];
        }
      }

      // Infer interval from sorted timestamps in the data
      const timestamps = Array.from(timeMap.keys()).sort((a, b) => a - b);
      if (timestamps.length < 2) return 60 * 60 * 1000; // Default to 1 hour

      // Calculate intervals between consecutive timestamps
      const intervals: number[] = [];
      for (let i = 1; i < timestamps.length; i++) {
        intervals.push(timestamps[i] - timestamps[i - 1]);
      }

      // Find the most common interval (mode)
      const intervalCounts = new Map<number, number>();
      intervals.forEach(interval => {
        intervalCounts.set(interval, (intervalCounts.get(interval) || 0) + 1);
      });

      let maxCount = 0;
      let mostCommonInterval = intervals[0];
      intervalCounts.forEach((count, interval) => {
        if (count > maxCount) {
          maxCount = count;
          mostCommonInterval = interval;
        }
      });

      return mostCommonInterval;
    })();

    // Step 3: Get sorted timestamps and determine time range
    const timestamps = Array.from(timeMap.keys()).sort((a, b) => a - b);
    if (timestamps.length === 0) return [];

    const minTime = timestamps[0];
    const maxTime = timestamps[timestamps.length - 1];

    // Step 4: Fill in all time buckets between min and max for uniform spacing
    // This ensures the chart displays evenly-spaced time points, with gaps shown as zeros
    // Safeguard: limit max buckets to prevent memory issues and UI freezing
    const MAX_BUCKETS = 1000;
    const timeRange = maxTime - minTime;
    let adjustedBucketMs = bucketMs;
    if (timeRange / adjustedBucketMs > MAX_BUCKETS) {
      adjustedBucketMs = Math.ceil(timeRange / MAX_BUCKETS);
    }

    const filledBuckets: Array<{ date: string; timestamp: number; counts: Record<string, number> }> = [];
    for (let t = minTime; t <= maxTime; t += adjustedBucketMs) {
      const counts = timeMap.get(t) || {};
      // Format date based on bucket size for better readability
      const date = new Date(t).toLocaleString([], {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
      filledBuckets.push({ date, timestamp: t, counts });
    }

    // Step 5: Transform to chart data format with all categories
    return filledBuckets
      .map(({ date, timestamp, counts }) => {
        const filledCounts: Record<string, number> = {};
        errorCategories.forEach(category => {
          filledCounts[category] = counts[category] || 0; // Zero for missing categories
        });
        return {
          date,
          timestamp,
          ...filledCounts
        } as Record<string, number> & { date: string; timestamp: number };
      });
  }, [data, errorCategories, timeBucketSize]);

  /**
   * Prepare ApexCharts series data from chartData.
   * Uses timestamp (x) and value (y) format for datetime x-axis.
   */
  const chartSeries = useMemo(() => {
    const categoriesToShow = selectedCategories.length > 0 ? selectedCategories : errorCategories;

    return categoriesToShow.map(category => ({
      name: category,
      data: chartData.map((d: Record<string, number> & { date: string; timestamp: number }) => ({
        x: d.timestamp,
        y: d[category] || 0,
      })),
    }));
  }, [chartData, selectedCategories, errorCategories]);

  /**
   * Handle chart zoom event - called when user zooms/drags to select a time range
   */
  const handleChartZoom = useCallback((_chartContext: any, { xaxis }: any) => {
    if (xaxis?.min !== undefined && xaxis?.max !== undefined && onTimeRangeSelect) {
      // Convert timestamps to datetime-local format (YYYY-MM-DDTHH:mm)
      const startDate = new Date(xaxis.min);
      const endDate = new Date(xaxis.max);

      const formatDateToLocal = (date: Date) => {
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        const hours = String(date.getHours()).padStart(2, '0');
        const minutes = String(date.getMinutes()).padStart(2, '0');
        return `${year}-${month}-${day}T${hours}:${minutes}`;
      };

      setIsCustomDate?.(true);
      onTimeRangeSelect(formatDateToLocal(startDate), formatDateToLocal(endDate));
      setCurrentPage?.(1);
    }
  }, [onTimeRangeSelect, setIsCustomDate, setCurrentPage]);

  /**
   * ApexCharts options configuration
   */
  const chartOptions = useMemo(() => {
    const categoriesToShow = selectedCategories.length > 0 ? selectedCategories : errorCategories;

    return {
      chart: {
        type: 'area' as const,
        height: 350,
        fontFamily: 'Inter, system-ui, sans-serif',
        toolbar: {
          show: false,
        },
        zoom: {
          enabled: true, 
          type: 'x',
          autoScaleYaxis: true,
          allowMouseWheelZoom: false,
        },
        animations: {
          enabled: true,
          easing: 'easeinout' as const,
          speed: 800,
        },
        events: {
          zoomed: onTimeRangeSelect ? handleChartZoom : undefined,
        },
      },
      colors: categoriesToShow.map(cat => categoryColors[cat]?.hex || '#888888'),
      dataLabels: {
        enabled: false,
      },
      stroke: {
        curve: 'smooth' as const,
        width: 2,
      },
      fill: {
        type: 'gradient' as const,
        gradient: {
          shadeIntensity: 1,
          opacityFrom: 0.7,
          opacityTo: 0.2,
          stops: [0, 90, 100],
        },
      },
      xaxis: {
        type: 'datetime' as const,
        labels: {
          rotate: -45,
          trim: true,
          maxHeight: 120,
          style: {
            colors: '#374151',
            fontSize: '11px',
            fontFamily: 'Inter, system-ui, sans-serif',
            fontWeight: 400,
          },
          datetimeUTC: false,
        },
        tooltip: {
          enabled: true,
        },
        axisBorder: {
          show: false,
        },
        axisTicks: {
          show: false,
        },
      },
      yaxis: {
        labels: {
          style: {
            colors: '#374151',
            fontSize: '11px',
            fontFamily: 'Inter, system-ui, sans-serif',
            fontWeight: 400,
          },
          formatter: (value: number) => value.toString(),
        },
      },
      grid: {
        borderColor: '#E5E7EB',
        strokeDashArray: 4,
        row: {
          colors: ['transparent'],
          opacity: 0.5,
        },
      },
      legend: {
        position: 'top' as const,
        horizontalAlign: 'left' as const,
        offsetY: 0,
      },
      tooltip: {
        theme: 'light' as const,
        x: {
          format: 'MMM dd, HH:mm',
        },
      },
    };
  }, [selectedCategories, errorCategories, categoryColors]);

  // Don't render if there's no data
  if (data.length === 0) {
    return null;
  }

  return (
    <Card className="mt-6">
      <Title>Failure Logs Aggregation</Title>

      {/* Header for the bar list */}
      <div className="flex justify-between mt-6 mb-2 px-2">
        <Text className="font-medium text-gray-500">Error Class</Text>
        <div className="flex items-center gap-4">
          {selectedCategories.length > 0 && (
            <button
              onClick={() => setSelectedCategories([])}
              className="text-xs text-blue-600 hover:text-blue-800 underline"
            >
              Clear Selection
            </button>
          )}
          <Text className="font-medium text-gray-500">Count</Text>
        </div>
      </div>

      {/* Interactive bar list showing error categories and their total counts */}
      {/* Clicking a bar toggles that category in the chart below */}
      <BarList
        data={errorStats}
        className="mt-2"
        onValueChange={(item) => {
          const name = item.name;
          setSelectedCategories(prev => {
            // If nothing is selected (all showing), select only this one
            if (prev.length === 0) {
              return [name];
            }
            // If this item is already selected, deselect it
            if (prev.includes(name)) {
              return prev.filter(c => c !== name);
            }
            // Otherwise, add this item to selection
            return [...prev, name];
          });
        }}
      />

      {/* Area chart showing failures over time using ApexCharts */}
      {chartData.length > 0 && chartSeries.length > 0 && (
        <div className="mt-8 border-t pt-6">
          <div className="flex items-center gap-2">
            <Title>Failures Over Time</Title>
            {/* Display the time bucket size for context */}
            {timeBucketSize && (
              <span className="text-sm text-gray-500 bg-gray-100 px-2 py-1 rounded">
                Time Bucket Size: {timeBucketSize}
              </span>
            )}
          </div>
          <div className="mt-4">
            <Chart
              options={chartOptions}
              series={chartSeries}
              type="area"
              height={350}
            />
          </div>
        </div>
      )}
    </Card>
  );
};