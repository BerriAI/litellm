import React from "react";
import { FormProvider, useForm } from "react-hook-form";

import {
  MountedFormProvider,
  useMountRegistry,
  type MountedFormValues,
} from "@/components/common_components/MountedFormField";

interface MountedFormHostProps {
  defaultValues?: MountedFormValues;
  children: React.ReactNode;
}

/**
 * Stands in for the page that owns the store, so a form child can be exercised on its own the way
 * an antd `<Form>` wrapper used to allow.
 */
export const MountedFormHost: React.FC<MountedFormHostProps> = ({ defaultValues, children }) => {
  const form = useForm<MountedFormValues>({ mode: "onChange", defaultValues });
  const registry = useMountRegistry();

  return (
    <FormProvider {...form}>
      <MountedFormProvider value={{ control: form.control, registry }}>{children}</MountedFormProvider>
    </FormProvider>
  );
};
