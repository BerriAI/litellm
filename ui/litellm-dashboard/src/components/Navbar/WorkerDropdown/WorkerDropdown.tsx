"use client";

import React from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useWorker } from "@/hooks/useWorker";

interface WorkerDropdownProps {
  onWorkerSwitch: (workerId: string) => void;
}

const WorkerDropdown: React.FC<WorkerDropdownProps> = ({ onWorkerSwitch }) => {
  const { isControlPlane, selectedWorker, workers } = useWorker();

  if (!isControlPlane || !selectedWorker) return null;

  return (
    <Select
      value={selectedWorker.worker_id}
      onValueChange={onWorkerSwitch}
    >
      <SelectTrigger className="min-w-[180px]">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {workers.map((w) => (
          <SelectItem
            key={w.worker_id}
            value={w.worker_id}
            disabled={w.worker_id === selectedWorker.worker_id}
          >
            {w.name}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
};

export default WorkerDropdown;
