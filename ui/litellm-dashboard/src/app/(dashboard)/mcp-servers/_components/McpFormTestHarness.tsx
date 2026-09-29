import * as React from "react";
import { render, type RenderResult } from "@testing-library/react";
import { FormProvider, useForm } from "react-hook-form";

import {
  MountedFormProvider,
  projectMountedValues,
  useMountRegistry,
  type MountedFormValues,
} from "@/components/common_components/MountedFormField";

export const McpFormHarness: React.FC<{
  defaultValues?: MountedFormValues;
  onFinish?: (values: MountedFormValues) => void;
  children: React.ReactNode;
}> = ({ defaultValues, onFinish, children }) => {
  const form = useForm<MountedFormValues>({ mode: "onChange", defaultValues });
  const registry = useMountRegistry();
  return (
    <FormProvider {...form}>
      <MountedFormProvider value={{ control: form.control, registry }}>
        <form
          onSubmit={(event) => {
            event.preventDefault();
            onFinish?.(projectMountedValues(registry, form.getValues));
          }}
        >
          {children}
          <button type="submit">Submit</button>
        </form>
      </MountedFormProvider>
    </FormProvider>
  );
};

export const renderInMcpForm = (ui: React.ReactNode, defaultValues: MountedFormValues = {}): RenderResult =>
  render(<McpFormHarness defaultValues={defaultValues}>{ui}</McpFormHarness>);
