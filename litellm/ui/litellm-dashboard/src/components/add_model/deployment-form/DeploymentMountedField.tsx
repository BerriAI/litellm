import { createContext, useContext, useLayoutEffect, useState, type PropsWithChildren } from "react";
import {
  MountedFormField,
  MountedFormProvider,
  type MountedFormContextValue,
  type MountedFormFieldProps,
  type MountedFormValues,
} from "@/components/common_components/MountedFormField";
import { useZodForm } from "@/lib/forms/useZodForm";
import { deploymentFormSchema, type FieldValidation } from "./schema";

const MountedValidationContext = createContext<Map<string, FieldValidation> | null>(null);
export const MountedValidationProvider = MountedValidationContext.Provider;

export function DeploymentMountedProvider({
  rules,
  value,
  children,
}: PropsWithChildren<{ rules: Map<string, FieldValidation> | null; value: MountedFormContextValue }>) {
  return (
    <MountedValidationProvider value={rules}>
      <MountedFormProvider value={value}>{children}</MountedFormProvider>
    </MountedValidationProvider>
  );
}

export function useDeploymentForm(defaultValues: MountedFormValues | undefined) {
  const [{ rules, schema }] = useState(() => {
    const rules = new Map<string, FieldValidation>();
    return { rules, schema: deploymentFormSchema(rules) };
  });
  const form = useZodForm(schema, { mode: "onChange", defaultValues });
  return { form, rules };
}

type Props = Omit<MountedFormFieldProps, "rules"> & {
  rules?: Pick<NonNullable<MountedFormFieldProps["rules"]>, "validate" | "deps">;
};

export function DeploymentMountedField(props: Props) {
  const mountedRules = useContext(MountedValidationContext);
  const validate = props.rules?.validate;
  const name = typeof props.name === "string" ? props.name : props.name.join(".");
  useLayoutEffect(() => {
    if (!mountedRules || !validate) return;
    mountedRules.set(name, validate);
    return () => {
      if (mountedRules.get(name) === validate) mountedRules.delete(name);
    };
  }, [mountedRules, name, validate]);
  return <MountedFormField {...props} />;
}
