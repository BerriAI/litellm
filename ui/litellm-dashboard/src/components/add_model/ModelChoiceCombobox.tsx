"use client";

import React from "react";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";

export interface ModelChoice {
  value: string;
  label: string;
}

interface ModelChoiceComboboxProps {
  id: string;
  value: string | null;
  onChange: (value: string | null) => void;
  choices: ModelChoice[];
  placeholder: string;
  ariaInvalid: true | undefined;
  ariaDescribedBy: string | undefined;
}

const ModelChoiceCombobox: React.FC<ModelChoiceComboboxProps> = ({
  id,
  value,
  onChange,
  choices,
  placeholder,
  ariaInvalid,
  ariaDescribedBy,
}) => {
  const selected = value ? choices.find((choice) => choice.value === value) ?? { value, label: value } : null;

  return (
    <Combobox
      items={choices}
      value={selected}
      onValueChange={(choice: ModelChoice | null) => onChange(choice?.value ?? null)}
      itemToStringLabel={(choice: ModelChoice) => choice.label}
      isItemEqualToValue={(choice: ModelChoice, current: ModelChoice) => choice.value === current.value}
    >
      <ComboboxInput
        id={id}
        aria-invalid={ariaInvalid}
        aria-describedby={ariaDescribedBy}
        placeholder={placeholder}
        className="w-full"
        showClear={value != null && value !== ""}
      />
      <ComboboxContent>
        <ComboboxEmpty>No models found</ComboboxEmpty>
        <ComboboxList>
          {(choice: ModelChoice) => (
            <ComboboxItem key={choice.value} value={choice}>
              {choice.label}
            </ComboboxItem>
          )}
        </ComboboxList>
      </ComboboxContent>
    </Combobox>
  );
};

export default ModelChoiceCombobox;
