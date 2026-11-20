import { useTranslation } from "react-i18next";
import { v4 as uuidv4 } from "uuid";
import { defaultCustomerCapPolicy, type CustomerBudgetGroupFields } from "./customerOrganizationForm";
import CustomerBudgetGroupsTable from "./CustomerBudgetGroupsTable";

export default function CustomerBudgetGroupsEditor({
  groups,
  defaultGroupId,
  onChange,
  disabled = false,
  memberGroups = true,
  appGroups = false,
}: {
  groups: CustomerBudgetGroupFields[];
  defaultGroupId: string;
  onChange: (groups: CustomerBudgetGroupFields[], defaultGroupId: string) => void;
  disabled?: boolean;
  memberGroups?: boolean;
  appGroups?: boolean;
}) {
  const { t } = useTranslation();
  const globalScope = appGroups ? "app" : "organization";
  const globalGroupKey = appGroups ? "customerPortal.appOrganizationGroups" : "customerPortal.organizationGroups";
  const updateGroup = (id: string, patch: Partial<Omit<CustomerBudgetGroupFields, "id">>) =>
    onChange(
      groups.map((group) => (group.id === id ? { ...group, ...patch } : group)),
      defaultGroupId,
    );
  const removeGroup = (id: string) => {
    const remaining = groups.filter((group) => group.id !== id);
    const nextDefault = memberGroups ? remaining[0]?.id ?? "" : "";
    onChange(remaining, defaultGroupId === id ? nextDefault : defaultGroupId);
  };
  const addGroup = () => {
    const group: CustomerBudgetGroupFields = {
      id: uuidv4(),
      name: "",
      monthlyBudget: "",
      yearlyBudget: "",
      tpm: "",
      rpm: "",
      capPolicy: defaultCustomerCapPolicy,
    };
    onChange([...groups, group], memberGroups && !groups.length ? group.id : defaultGroupId);
  };
  return (
    <CustomerBudgetGroupsTable
      groups={groups}
      defaultGroupId={defaultGroupId}
      disabled={disabled}
      scope={memberGroups ? "member" : globalScope}
      title={t(memberGroups ? "customerPortal.createDialog.memberGroups" : `${globalGroupKey}.title`)}
      description={t(memberGroups ? "customerPortal.createDialog.memberGroupsHelp" : `${globalGroupKey}.help`)}
      onUpdate={updateGroup}
      onRemove={removeGroup}
      onAdd={addGroup}
      onDefaultChange={(id) => onChange(groups, id)}
    />
  );
}
