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

export const MountedFormHost: React.FC<MountedFormHostProps> = ({ defaultValues, children }) => {
  const form = useForm<MountedFormValues>({ mode: "onChange", defaultValues });
  const registry = useMountRegistry();

  return (
    <FormProvider {...form}>
      <MountedFormProvider value={{ control: form.control, registry }}>{children}</MountedFormProvider>
    </FormProvider>
  );
};
