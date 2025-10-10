import React from "react";
import { UiLoadingSpinner } from "../ui/ui-loading-spinner";

interface ChartLoaderProps {
  isDateChanging?: boolean;
}

export const ChartLoader: React.FC<ChartLoaderProps> = ({ isDateChanging = false }) => (
  <div className="flex items-center justify-center h-40">
    <div className="flex items-center justify-center gap-3">
      <UiLoadingSpinner className="size-5" />
      <div className="flex flex-col">
        <span className="text-gray-600 text-sm font-medium">
          {isDateChanging ? "Processing date selection..." : "Loading chart data..."}
        </span>
        <span className="text-gray-400 text-xs mt-1">
          {isDateChanging ? "This will only take a moment" : "Fetching your data"}
        </span>
      </div>
    </div>
  </div>
);

export default ChartLoader;
