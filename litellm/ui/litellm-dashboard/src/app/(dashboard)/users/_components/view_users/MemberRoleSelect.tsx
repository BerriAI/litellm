import type { ParseKeys } from "i18next";
import { useTranslation } from "react-i18next";
import { Field, FieldLabel } from "@/components/ui/field";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { SimpleTooltip } from "@/components/ui/tooltip";

const ADD_TEAM_ROLE_FIELD_ID = "add-team-role";

const MEMBER_ROLE_OPTIONS: ReadonlyArray<{ value: string; hintKey: ParseKeys }> = [
  { value: "user", hintKey: "users.userInfo.memberRoles.userHint" },
  { value: "admin", hintKey: "users.userInfo.memberRoles.adminHint" },
];

export function MemberRoleSelect({ value, onChange }: { value: string; onChange: (value: string) => void }) {
  const { t } = useTranslation();
  return (
    <Field>
      <FieldLabel htmlFor={ADD_TEAM_ROLE_FIELD_ID}>{t("users.userInfo.addTeamDialog.memberRole")}</FieldLabel>
      <Select value={value} onValueChange={(value) => value !== null && onChange(value)}>
        <SelectTrigger id={ADD_TEAM_ROLE_FIELD_ID} className="w-full">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {MEMBER_ROLE_OPTIONS.map((option) => (
            <SelectItem key={option.value} value={option.value} title={option.value}>
              <SimpleTooltip content={t(option.hintKey)}>
                <span className="font-medium">{option.value}</span>
                <span className="ml-2 text-muted-foreground text-sm">- {t(option.hintKey)}</span>
              </SimpleTooltip>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </Field>
  );
}
