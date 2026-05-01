"use client";

import React from "react";
import { Select } from "antd";
import { CloudServerOutlined } from "@ant-design/icons";
import { useWorker } from "@/hooks/useWorker";

interface WorkerDropdownProps {
  onWorkerSwitch: (workerId: string) => void;
}

const WorkerDropdown: React.FC<WorkerDropdownProps> = ({ onWorkerSwitch }) => {
  const { isControlPlane, selectedWorker, workers } = useWorker();

  if (!isControlPlane || !selectedWorker) return null;

  return (
    <Select
      showSearch
      filterOption={(input, option) =>
        (option?.label as string ?? "").toLowerCase().includes(input.toLowerCase())
      }
      value={selectedWorker.worker_id}
      style={{ minWidth: 180 }}
      suffixIcon={<CloudServerOutlined />}
      options={workers.map((w) => ({
        label: w.name,
        value: w.worker_id,
        disabled: w.worker_id === selectedWorker.worker_id,
      }))}
      onChange={(newWorkerId) => {
        onWorkerSwitch(newWorkerId);
      }}
    />
  );
};

export default WorkerDropdown;
