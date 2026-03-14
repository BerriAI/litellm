import { BarChart, Card, Title } from "@tremor/react";
import React from "react";

const data = [
  { date: "2026-03-05", healthy: 120, drifting: 10, rogue: 2 },
  { date: "2026-03-06", healthy: 132, drifting: 15, rogue: 1 },
  { date: "2026-03-07", healthy: 145, drifting: 12, rogue: 3 },
  { date: "2026-03-08", healthy: 150, drifting: 8, rogue: 0 },
  { date: "2026-03-09", healthy: 148, drifting: 20, rogue: 5 },
  { date: "2026-03-10", healthy: 160, drifting: 18, rogue: 2 },
  { date: "2026-03-11", healthy: 165, drifting: 14, rogue: 4 },
];

export const AgentHealthChart: React.FC = () => (
  <Card className="bg-white border border-gray-200 h-full flex flex-col">
    <div className="flex justify-between items-center mb-4">
      <Title className="text-base font-semibold text-gray-900">
        Agent Outcomes Over Time
      </Title>
      <div className="flex items-center space-x-4 text-sm">
        <div className="flex items-center">
          <div className="w-2 h-2 rounded-full bg-emerald-500 mr-2" />
          <span className="text-gray-600">healthy</span>
        </div>
        <div className="flex items-center">
          <div className="w-2 h-2 rounded-full bg-amber-500 mr-2" />
          <span className="text-gray-600">drifting</span>
        </div>
        <div className="flex items-center">
          <div className="w-2 h-2 rounded-full bg-red-500 mr-2" />
          <span className="text-gray-600">rogue</span>
        </div>
      </div>
    </div>

    <div className="flex-1 min-h-[250px]">
      <BarChart
        data={data}
        index="date"
        categories={["healthy", "drifting", "rogue"]}
        colors={["emerald", "amber", "red"]}
        valueFormatter={(v) => v.toLocaleString()}
        yAxisWidth={48}
        showLegend={false}
        stack={true}
      />
    </div>
  </Card>
);
