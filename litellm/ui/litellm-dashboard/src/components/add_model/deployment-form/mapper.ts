import {
  projectMountedValues,
  type MountRegistry,
  type MountedFormValues,
} from "@/components/common_components/MountedFormField";
import type { UseFormGetValues } from "react-hook-form";

export const deploymentDefaults: MountedFormValues = { litellm_credential_name: null };

export function readDeploymentDraft(registry: MountRegistry, getValues: UseFormGetValues<MountedFormValues>) {
  return projectMountedValues(registry, getValues);
}
