import React from "react";
import { CircleHelp } from "lucide-react";

import { Logo } from "@/components/molecules/logo/Logo";
import { Providers } from "@/components/provider_info_helpers";
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field";
import { Button } from "@/components/ui/button";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import { Input } from "@/components/ui/input";
import { InputGroupAddon } from "@/components/ui/input-group";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

interface ProviderOption {
  value: string;
  label: string;
}

interface AddProviderFormProps {
  selectedProvider: string | undefined;
  newDiscount: string;
  modelPattern: string;
  onProviderChange: (provider: string | undefined) => void;
  onDiscountChange: (discount: string) => void;
  onModelPatternChange: (value: string) => void;
  onAddProvider: () => void;
}

const PROVIDER_FIELD_ID = "add-provider-discount-provider";
const MODEL_PATTERN_FIELD_ID = "add-provider-discount-model-pattern";
const DISCOUNT_FIELD_ID = "add-provider-discount-percentage";

const providerOptions = (): ProviderOption[] => Object.entries(Providers).map(([value, label]) => ({ value, label }));

const selectedProviderOption = (selectedProvider: string | undefined): ProviderOption | null => {
  if (!selectedProvider) {
    return null;
  }
  const label = Providers[selectedProvider as keyof typeof Providers];
  return label ? { value: selectedProvider, label } : null;
};

const labelWithHint = (label: string, hint: string): React.ReactNode => (
  <>
    {label}
    <Tooltip>
      <TooltipTrigger render={<CircleHelp className="size-3.5 shrink-0 cursor-help text-muted-foreground" />} />
      <TooltipContent>{hint}</TooltipContent>
    </Tooltip>
  </>
);

const AddProviderForm: React.FC<AddProviderFormProps> = ({
  selectedProvider,
  newDiscount,
  modelPattern,
  onProviderChange,
  onDiscountChange,
  onModelPatternChange,
  onAddProvider,
}) => {
  const options = providerOptions();
  const selectedOption = selectedProviderOption(selectedProvider);

  return (
    <TooltipProvider>
      <div className="space-y-6">
        <FieldGroup>
          <Field>
            <FieldLabel htmlFor={PROVIDER_FIELD_ID}>
              {labelWithHint("Provider", "Select the LLM provider you want to configure a discount for")}
            </FieldLabel>
            <Combobox
              items={options}
              value={selectedOption}
              onValueChange={(option: ProviderOption | null) => onProviderChange(option?.value)}
              itemToStringLabel={(option: ProviderOption) => option.label}
              isItemEqualToValue={(option: ProviderOption, value: ProviderOption) => option.value === value.value}
            >
              <ComboboxInput id={PROVIDER_FIELD_ID} placeholder="Select provider" className="w-full">
                {selectedOption && (
                  <InputGroupAddon align="inline-start">
                    <Logo provider={selectedOption.value} label={selectedOption.label} className="w-5 h-5" />
                  </InputGroupAddon>
                )}
              </ComboboxInput>
              <ComboboxContent>
                <ComboboxEmpty>No providers found</ComboboxEmpty>
                <ComboboxList>
                  {(option: ProviderOption) => (
                    <ComboboxItem key={option.value} value={option}>
                      <span className="flex items-center space-x-2">
                        <Logo provider={option.value} label={option.label} className="w-5 h-5" />
                        <span>{option.label}</span>
                      </span>
                    </ComboboxItem>
                  )}
                </ComboboxList>
              </ComboboxContent>
            </Combobox>
          </Field>

          <Field>
            <FieldLabel htmlFor={MODEL_PATTERN_FIELD_ID}>
              {labelWithHint(
                "Model pattern",
                "Leave empty to discount every model from this provider. Use a glob such as claude-* or gemini-2* to discount only matching models",
              )}
            </FieldLabel>
            <Input
              id={MODEL_PATTERN_FIELD_ID}
              placeholder="claude-*"
              value={modelPattern}
              onChange={(event) => onModelPatternChange(event.target.value)}
              className="rounded-lg"
            />
          </Field>

          <Field>
            <FieldLabel htmlFor={DISCOUNT_FIELD_ID}>
              {labelWithHint("Discount Percentage", "Enter a percentage value (e.g., 5 for 5% discount)")}
            </FieldLabel>
            <div className="flex items-center gap-2">
              <Input
                id={DISCOUNT_FIELD_ID}
                placeholder="5"
                value={newDiscount}
                onChange={(event) => onDiscountChange(event.target.value)}
                className="flex-1 rounded-lg"
              />
              <span className="text-muted-foreground">%</span>
            </div>
          </Field>
        </FieldGroup>

        <div className="flex items-center justify-end space-x-3 pt-6 border-t border-border">
          <Button type="submit" onClick={onAddProvider} disabled={!selectedProvider || !newDiscount}>
            Add Provider Discount
          </Button>
        </div>
      </div>
    </TooltipProvider>
  );
};

export default AddProviderForm;
