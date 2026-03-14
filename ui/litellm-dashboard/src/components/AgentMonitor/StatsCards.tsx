import { Card, Text, Title } from "@tremor/react";
import { AlertTriangle, CheckCircle2, Power } from "lucide-react";
import React from "react";

interface StatCardProps {
  label: string;
  value: string | number;
  valueColor?: string;
  icon?: React.ReactNode;
}

const StatCard: React.FC<StatCardProps> = ({
  label,
  value,
  valueColor = "text-gray-900",
  icon,
}) => (
  <Card className="bg-white border border-gray-200 p-5">
    <div className="flex justify-between items-start mb-2">
      <Text className="text-sm font-medium text-gray-500">{label}</Text>
      {icon && <div>{icon}</div>}
    </div>
    <Title className={`text-3xl font-bold ${valueColor}`}>{value}</Title>
  </Card>
);

export const StatsCards: React.FC = () => (
  <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-6">
    <StatCard label="Monitored Agents" value={24} />
    <StatCard
      label="Drift Alerts"
      value={3}
      icon={<AlertTriangle className="h-4 w-4 text-amber-500" />}
    />
    <StatCard
      label="Pass Rate"
      value="96%"
      valueColor="text-emerald-500"
      icon={<CheckCircle2 className="h-4 w-4 text-emerald-500" />}
    />
    <StatCard label="Avg. Detection Latency" value="12ms" valueColor="text-emerald-500" />
    <StatCard
      label="Active Kill Switches"
      value={2}
      icon={<Power className="h-4 w-4 text-gray-400" />}
    />
  </div>
);
