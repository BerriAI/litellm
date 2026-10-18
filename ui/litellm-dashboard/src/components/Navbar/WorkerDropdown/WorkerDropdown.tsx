"use client";

import React from "react";
import { Server } from "lucide-react";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import { InputGroupAddon } from "@/components/ui/input-group";
import { useWorker } from "@/hooks/useWorker";

interface WorkerDropdownProps {
  onWorkerSwitch: (workerId: string) => void;
}

interface WorkerOption {
  label: string;
  value: string;
  disabled: boolean;
}

const WorkerDropdown: React.FC<WorkerDropdownProps> = ({ onWorkerSwitch }) => {
  const { isControlPlane, selectedWorker, workers } = useWorker();

  if (!isControlPlane || !selectedWorker) return null;

  const options: WorkerOption[] = workers.map((w) => ({
    label: w.name,
    value: w.worker_id,
    disabled: w.worker_id === selectedWorker.worker_id,
  }));

  return (
    <Combobox
      items={options}
      value={options.find((option) => option.value === selectedWorker.worker_id) ?? null}
      itemToStringLabel={(option: WorkerOption) => option.label}
      onValueChange={(option: WorkerOption | null) => {
        if (option) {
          onWorkerSwitch(option.value);
        }
      }}
    >
      <ComboboxInput className="min-w-[180px]" aria-label="Worker">
        <InputGroupAddon align="inline-start">
          <Server className="size-4" />
        </InputGroupAddon>
      </ComboboxInput>
      <ComboboxContent>
        <ComboboxEmpty>No matching workers</ComboboxEmpty>
        <ComboboxList>
          {(option: WorkerOption) => (
            <ComboboxItem key={option.value} value={option} disabled={option.disabled}>
              {option.label}
            </ComboboxItem>
          )}
        </ComboboxList>
      </ComboboxContent>
    </Combobox>
  );
};

export default WorkerDropdown;
