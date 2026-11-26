import { type FormEvent } from "react";
import { useWatch } from "react-hook-form";
import { useTranslation } from "react-i18next";
import { useZodForm } from "@/lib/forms/useZodForm";
import { useCustomerBudgetSave } from "../useCustomerBudgetSave";
import { useSaveCustomerHierarchy } from "../useCustomerHierarchyForm";
import {
  CUSTOMER_HIERARCHY_FORM_ERRORS,
  type CustomerHierarchyEntity,
  type CustomerHierarchyFields,
  type Organization,
} from "../customerHierarchyForms";
import { hierarchyEditDefaults } from "./mapper";
import { hierarchyEditSchema } from "./schema";

export function useHierarchySettingsForm(
  entity: CustomerHierarchyEntity,
  organization: Organization,
  onSaved: () => void,
) {
  const { t } = useTranslation();
  const initial = hierarchyEditDefaults(entity, organization);
  const schema = hierarchyEditSchema(entity, organization);
  const { control, getValues, reset, setValue } = useZodForm(schema, { defaultValues: initial });
  const [alias, budget] = useWatch({ control, name: ["alias", "budget"] });
  const { pending, error, setError, run } = useCustomerBudgetSave();
  const write = useSaveCustomerHierarchy(entity, onSaved);
  const updateBudget = <K extends keyof CustomerHierarchyFields>(key: K, value: CustomerHierarchyFields[K]) => {
    setValue("budget", { ...getValues("budget"), [key]: value });
    setError(null);
  };
  const resetChanges = () => {
    reset(initial);
    setError(null);
  };
  const save = async (event: FormEvent) => {
    event.preventDefault();
    if (pending) return;
    const result = schema.parse(getValues());
    if (!result.ok) return setError(t(CUSTOMER_HIERARCHY_FORM_ERRORS[result.error]));
    await run(() => write(result.patch, () => setError(t("customerPortal.hierarchy.error"))));
  };
  const dirty = JSON.stringify({ alias, budget }) !== JSON.stringify(initial);
  return { control, alias, updateBudget, resetChanges, pending, error, setError, save, dirty };
}
