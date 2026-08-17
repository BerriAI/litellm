import React from "react";
import { useFormContext } from "react-hook-form";

import { Field, FieldError, FieldLabel } from "@/components/shared/form/field";
import { Input } from "@/components/ui/input";
import { CredentialAccess } from "../Settings/LoggingAndAlerts/LoggingCallbacks/types";
import AccessControlFields from "./AccessControlFields";
import { LoggingField } from "./loggingDestinationFields";

type DestinationFormValues = Record<string, string>;

interface DestinationFieldsProps {
  fields: readonly LoggingField[];
  access: CredentialAccess;
  onAccessChange: (access: CredentialAccess) => void;
}

const DestinationFields: React.FC<DestinationFieldsProps> = ({ fields, access, onAccessChange }) => {
  const { register, formState } = useFormContext<DestinationFormValues>();
  const fieldIdPrefix = React.useId();

  return (
    <div className="space-y-4 mt-6 p-4 bg-gray-50 rounded-lg border">
      <Field>
        <FieldLabel htmlFor={`${fieldIdPrefix}-credential_name`}>
          <span className="text-sm font-medium text-gray-700">Name</span>
        </FieldLabel>
        <Input
          id={`${fieldIdPrefix}-credential_name`}
          placeholder="e.g. langfuse-eu"
          {...register("credential_name", { required: "Please enter a name" })}
        />
        <FieldError errors={[formState.errors.credential_name]} />
      </Field>
      {fields.map((f) => (
        <Field key={f.name}>
          <FieldLabel htmlFor={`${fieldIdPrefix}-${f.name}`}>
            <span className="text-sm font-medium text-gray-700">{f.label}</span>
          </FieldLabel>
          <Input
            id={`${fieldIdPrefix}-${f.name}`}
            type={f.type === "password" ? "password" : "text"}
            placeholder={f.placeholder}
            {...register(f.name, f.optional ? undefined : { required: `Please enter the ${f.label.toLowerCase()}` })}
          />
          <FieldError errors={[formState.errors[f.name]]} />
        </Field>
      ))}
      <AccessControlFields value={access} onChange={onAccessChange} />
    </div>
  );
};

export default DestinationFields;
