import type { MakeModelPublicValues } from "./makeModelPublicSchema";

export const toModelPublicationRequest = ({ modelGroups }: MakeModelPublicValues) => ({
  model_groups: modelGroups,
});
