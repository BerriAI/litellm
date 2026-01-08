import React from "react";
import EndpointUsageBarChart from "./components/EndpointUsageBarChart";
import EndpointUsageLineChart from "./components/EndpointUsageLineChart";
import EndpointUsageTable from "./components/EndpointUsageTable";
import { mockEndpointData } from "./mockEndpointData";
import { DailyData } from "../../types";

interface EndpointUsageProps {
  userSpendData?: {
    results: DailyData[];
    metadata: any;
  };
}

const EndpointUsage: React.FC<EndpointUsageProps> = ({ userSpendData }) => {
  return (
    <div className="space-y-4">
      <EndpointUsageTable endpointData={mockEndpointData} />
      <EndpointUsageBarChart endpointData={mockEndpointData} />
      <EndpointUsageLineChart dailyData={userSpendData} endpointData={mockEndpointData} />
    </div>
  );
};

export default EndpointUsage;
