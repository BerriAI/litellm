import TeamDropdown from "@/components/common_components/team_dropdown";
import { requiredRule } from "@/components/common_components/formRules";
import { MountedFormField } from "@/components/common_components/MountedFormField";
import { labelWithHint } from "@/components/shared/form/LabelWithHint";
import { Input } from "@/components/ui/input";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";

import { CHATGPT_OWNER_FIELDS, type ChatGPTOwnerKind } from "./chatgpt_sign_in_flow";

const OWNER_KIND_LABEL_ID = "chatgpt_owner_kind_label";

interface ChatGPTOwnerFieldsProps {
  readonly ownerKind: ChatGPTOwnerKind;
  readonly onOwnerKindChange: (ownerKind: ChatGPTOwnerKind) => void;
  readonly storedOwnerTeamId: string | null;
  readonly storedOwnerKeyHash: string | null;
}

export default function ChatGPTOwnerFields({
  ownerKind,
  onOwnerKindChange,
  storedOwnerTeamId,
  storedOwnerKeyHash,
}: ChatGPTOwnerFieldsProps) {
  return (
    <>
      <div className="mb-4 flex flex-col gap-2">
        <span id={OWNER_KIND_LABEL_ID} className="text-sm font-medium">
          {labelWithHint("Owner:", "Only the team or virtual key you pick can use this ChatGPT subscription.")}
        </span>
        <RadioGroup
          aria-labelledby={OWNER_KIND_LABEL_ID}
          value={ownerKind}
          onValueChange={(value) => onOwnerKindChange(value as ChatGPTOwnerKind)}
          className="flex w-auto items-center gap-4"
        >
          <label className="flex items-center gap-2 text-sm">
            <RadioGroupItem value="team" />
            Team
          </label>
          <label className="flex items-center gap-2 text-sm">
            <RadioGroupItem value="key" />
            Virtual key
          </label>
        </RadioGroup>
      </div>
      {ownerKind === "team" ? (
        <MountedFormField
          key={CHATGPT_OWNER_FIELDS.team}
          label="Team:"
          name={CHATGPT_OWNER_FIELDS.team}
          required
          defaultValue={storedOwnerTeamId}
          rules={{ validate: { required: requiredRule("Pick the team that owns this login") } }}
          className="mb-4"
        >
          {(control) => (
            <TeamDropdown
              id={control.id}
              value={typeof control.value === "string" ? control.value : null}
              onChange={control.onChange}
            />
          )}
        </MountedFormField>
      ) : (
        <MountedFormField
          key={CHATGPT_OWNER_FIELDS.key}
          label="Key ID:"
          name={CHATGPT_OWNER_FIELDS.key}
          required
          defaultValue={storedOwnerKeyHash}
          rules={{ validate: { required: requiredRule("Enter the ID of the key that owns this login") } }}
          className="mb-4"
        >
          {(control) => (
            <Input
              id={control.id}
              value={typeof control.value === "string" ? control.value : ""}
              onChange={control.onChange}
              onBlur={control.onBlur}
              placeholder="The Key ID shown on the Virtual Keys page"
            />
          )}
        </MountedFormField>
      )}
    </>
  );
}
