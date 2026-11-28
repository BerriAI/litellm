import type { CreateUserFormValues } from "./schema";

export const buildCreateUserPayload = (values: CreateUserFormValues, isPersonalKeyOpen: boolean) => {
  const { supply_user_id, user_id, ...fields } = values;
  if (!isPersonalKeyOpen) {
    delete fields.models;
  }
  const apiValues = supply_user_id ? { ...fields, user_id: user_id?.trim() } : fields;
  const withModels =
    (!fields.models || fields.models.length === 0) && values.user_role !== "proxy_admin"
      ? { ...apiValues, models: ["no-default-models"] }
      : apiValues;
  if (!withModels.organization_ids) {
    return withModels;
  }
  const { organization_ids, ...rest } = withModels;
  return { ...rest, organizations: organization_ids };
};
