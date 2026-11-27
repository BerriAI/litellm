import { useContext, useEffect, type ComponentProps, type ContextType, type ReactNode } from "react";
import { FormProvider, type UseFormReturn } from "react-hook-form";

import { MountedFormField, type MountedFormValues } from "@/components/common_components/MountedFormField";

import { McpValidationContext } from "./schema";

export const McpFormField = (props: ComponentProps<typeof MountedFormField>) => {
  const registerValidation = useContext(McpValidationContext);
  const name = typeof props.name === "string" ? props.name : props.name.join(".");
  const validate = props.rules?.validate;
  useEffect(() => (validate ? registerValidation?.(name, validate) : undefined), [name, registerValidation, validate]);
  return <MountedFormField {...props} rules={registerValidation ? undefined : props.rules} />;
};

export const McpFormProvider = ({
  form,
  registerValidation,
  children,
}: {
  form: UseFormReturn<MountedFormValues>;
  registerValidation: NonNullable<ContextType<typeof McpValidationContext>>;
  children: ReactNode;
}) => (
  <McpValidationContext.Provider value={registerValidation}>
    <FormProvider {...form}>{children}</FormProvider>
  </McpValidationContext.Provider>
);
