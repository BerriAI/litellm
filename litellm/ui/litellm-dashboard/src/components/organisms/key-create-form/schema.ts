import { formatNumberWithCommas } from "@/utils/dataUtils";
import type { TFunction } from "i18next";
import { z } from "zod";
import type { MountRegistry } from "../../common_components/MountedFormField";

type TeamCeilings = Partial<Record<"max_budget" | "tpm_limit" | "rpm_limit" | "tpd_limit", number | null>>;

interface KeyValidationContext {
  keyOwner: string;
  ownerRequiresTeam: boolean;
  ownerLimitedToTeamMembers: boolean;
  team: TeamCeilings | null | undefined;
}

export const createKeySchema = (
  registry: MountRegistry,
  { keyOwner, ownerRequiresTeam, ownerLimitedToTeamMembers, team }: KeyValidationContext,
  t: TFunction,
) =>
  z.record(z.string(), z.unknown()).superRefine((values, ctx) => {
    const mounted = new Set(registry.mountedNames());
    const requireField = (name: string, required: boolean, message: string) => {
      const value = values[name];
      const blank = value === undefined || value === null || value === "";
      if (required && mounted.has(name) && blank) {
        ctx.addIssue({ code: "custom", path: [name], message });
      }
    };
    requireField("user_id", keyOwner === "another_user", t("createKey.userId.required"));
    requireField(
      "team_id",
      ownerRequiresTeam,
      t(ownerLimitedToTeamMembers ? "createKey.team.requiredMemberTeam" : "createKey.team.requiredServiceAccount"),
    );
    requireField(
      "key_alias",
      true,
      t("createKey.keyName.required", {
        name: t(
          keyOwner === "you" ? "createKey.keyName.requiredNames.keyName" : "createKey.keyName.requiredNames.serviceAccountId",
        ),
      }),
    );

    const ceilings = {
      max_budget: "createKey.maxBudget.ceiling",
      tpm_limit: "createKey.tpm.ceiling",
      rpm_limit: "createKey.rpm.ceiling",
      tpd_limit: "createKey.tpd.ceiling",
    } as const;
    for (const name of Object.keys(ceilings) as (keyof typeof ceilings)[]) {
      const limit = team?.[name];
      const aboveLimit = limit != null && values[name] && (values[name] as number) > limit;
      if (mounted.has(name) && aboveLimit) {
        const message = t(ceilings[name], { limit: name === "max_budget" ? formatNumberWithCommas(limit, 4) : limit });
        ctx.addIssue({ code: "custom", path: [name], message });
      }
    }
  });
