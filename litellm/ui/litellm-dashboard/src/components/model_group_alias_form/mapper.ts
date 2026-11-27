import type { AliasItem } from "../model_group_alias_ordering";
import type { ModelGroupAliasFields } from "./schema";

export const emptyAlias: ModelGroupAliasFields = { aliasName: "", targetModelGroup: "" };

export const aliasToFields = ({ aliasName, targetModelGroup }: AliasItem): ModelGroupAliasFields => ({
  aliasName,
  targetModelGroup,
});

export const aliasToItem = (id: string, values: ModelGroupAliasFields): AliasItem => ({ id, ...values });
