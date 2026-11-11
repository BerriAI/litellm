"use client";

import React from "react";
import { useTranslation } from "react-i18next";
import { Server } from "lucide-react";
import { Combobox as ComboboxPrimitive } from "@base-ui/react/combobox";
import { Button } from "@/components/ui/button";
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
  compact?: boolean;
}

interface WorkerOption {
  label: string;
  value: string;
  disabled: boolean;
}

const WorkerDropdown: React.FC<WorkerDropdownProps> = ({ onWorkerSwitch, compact = false }) => {
  const { t } = useTranslation();
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
      {compact ? (
        <ComboboxPrimitive.Trigger
          render={<Button variant="ghost" size="icon" className="text-muted-foreground" />}
          aria-label={t("shell.workerDropdown.trigger", { worker: selectedWorker.name })}
          title={selectedWorker.name}
        >
          <Server className="size-4" aria-hidden />
        </ComboboxPrimitive.Trigger>
      ) : (
        <ComboboxInput className="min-w-[180px]" aria-label={t("shell.workerDropdown.label")}>
          <InputGroupAddon align="inline-start">
            <Server className="size-4" />
          </InputGroupAddon>
        </ComboboxInput>
      )}
      <ComboboxContent align={compact ? "end" : "start"} className={compact ? "w-64" : undefined}>
        {compact && (
          <ComboboxInput
            aria-label={t("shell.workerDropdown.searchLabel")}
            placeholder={t("shell.workerDropdown.searchPlaceholder")}
            showTrigger={false}
          />
        )}
        <ComboboxEmpty>{t("shell.workerDropdown.empty")}</ComboboxEmpty>
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
