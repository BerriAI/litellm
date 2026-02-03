import React, { useState, useMemo } from "react";
import { Card, Title, BarList, Text, AreaChart } from "@tremor/react";

interface ErrorStat {
  time_bucket: string;
  extracted_error: string;
  count: number;
}

interface ErrorStatsTableProps {
  data: ErrorStat[];
  timeBucketSize?: string;
}

const COLORS = [
  "red", "orange", "amber", "yellow", "lime", "green", "emerald", "teal", "cyan", "sky", "blue", "indigo", "violet", "purple", "fuchsia", "pink", "rose",
  "slate", "gray", "zinc", "neutral", "stone",
];

export const ErrorStatsTable: React.FC<ErrorStatsTableProps> = ({ data, timeBucketSize }) => {
  const [selectedCategories, setSelectedCategories] = useState<string[]>([]);

  const errorCategories = useMemo(() => {
    const uniqueErrors = new Set(data.map(item => item.extracted_error || "Unknown Error"));
    return Array.from(uniqueErrors);
  }, [data]);

  const categoryColors = useMemo(() => {
    const mapping: Record<string, string> = {};
    errorCategories.forEach((cat, index) => {
      mapping[cat] = COLORS[index % COLORS.length];
    });
    return mapping;
  }, [errorCategories]);

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
          ? categoryColors[name]
          : "gray"
      }))
      .sort((a, b) => b.value - a.value);
  }, [data, selectedCategories, categoryColors]);

  const chartData = useMemo(() => {
    if (data.length === 0) return [];

    const timeMap = new Map<string, { timestamp: number; counts: Record<string, number> }>();

    data.forEach((item) => {
      if (!item.time_bucket) return;
      try {
        const date = new Date(item.time_bucket);
        const key = date.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
        const timestamp = date.getTime();

        if (!timeMap.has(key)) {
          timeMap.set(key, { timestamp, counts: {} });
        }

        const errorClass = item.extracted_error || "Unknown Error";
        const bucket = timeMap.get(key)!;
        bucket.counts[errorClass] = (bucket.counts[errorClass] || 0) + item.count;
      } catch (e) {
        console.error("Error parsing date:", e);
      }
    });

    return Array.from(timeMap.entries())
      .map(([date, { timestamp, counts }]) => {
        const filledCounts: Record<string, number> = {};
        errorCategories.forEach(category => {
          filledCounts[category] = counts[category] || 0;
        });
        return {
          date,
          ...filledCounts,
          timestamp
        };
      })
      .sort((a, b) => a.timestamp - b.timestamp)
      .map(({ timestamp, ...rest }) => rest);
  }, [data, errorCategories]);

  if (data.length === 0) {
    return null;
  }

  return (
    <Card className="mt-6">
      <Title>Failure Logs Aggregation</Title>

      <div className="flex justify-between mt-6 mb-2 px-2">
        <Text className="font-medium text-gray-500">Error Class</Text>
        <Text className="font-medium text-gray-500">Count</Text>
      </div>

      <BarList
        data={errorStats}
        className="mt-2"
        onValueChange={(item) => {
          const name = item.name;
          setSelectedCategories(prev => {
            if (prev.includes(name)) {
              return prev.filter(c => c !== name);
            } else {
              return [...prev, name];
            }
          });
        }}
      />

      {chartData.length > 0 && (
        <div className="mt-8 border-t pt-6">
          <div className="flex items-center gap-2">
            <Title>Failures Over Time</Title>
            {timeBucketSize && (
              <span className="text-sm text-gray-500 bg-gray-100 px-2 py-1 rounded">
                Time Bucket Size: {timeBucketSize}
              </span>
            )}
          </div>
          <AreaChart
            className="h-72 mt-4"
            data={chartData}
            index="date"
            categories={selectedCategories.length > 0 ? selectedCategories : errorCategories}
            colors={(selectedCategories.length > 0 ? selectedCategories : errorCategories).map(cat => categoryColors[cat])}
            yAxisWidth={40}
            showAnimation={true}
            showXAxis={true}
          />
        </div>
      )}
    </Card>
  );
};