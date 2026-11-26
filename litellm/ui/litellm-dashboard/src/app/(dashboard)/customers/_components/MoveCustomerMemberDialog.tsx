"use client";

import { useState, type FormEvent } from "react";
import { Controller } from "react-hook-form";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { fetchClient } from "@/lib/http/api";
import { useZodForm } from "@/lib/forms/useZodForm";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { toast } from "sonner";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import type { Team, Member } from "./customerHierarchyForms";
import { memberMoveSchema } from "./customer-member-move/schema";

async function teamChoices(organizationId: string, signal: AbortSignal, page = 1): Promise<Team[]> {
  if (page > 1_000) throw new Error("Customer team list is incomplete");
  const response = await fetchClient.GET("/management/v1/customer_teams", {
    params: { query: { parent_org_id: organizationId, page, page_size: 100 } },
    signal,
  });
  if (!response.data) throw new Error("Missing customer teams response");
  if (page >= response.data.meta.total_pages) return response.data.data;
  return [...response.data.data, ...(await teamChoices(organizationId, signal, page + 1))];
}

function CustomerMemberTeams({
  organizationId,
  value,
  onChange,
}: {
  organizationId: string;
  value: readonly string[];
  onChange: (value: string[]) => void;
}) {
  const { t } = useTranslation();
  const teams = useQuery({
    queryKey: ["customer-hierarchy", "team-choices", organizationId],
    queryFn: ({ signal }) => teamChoices(organizationId, signal),
  });
  const options = (teams.data ?? []).map((team) => ({
    value: team.customer_team_id,
    label: team.alias || team.customer_team_id,
    description: team.alias ? team.customer_team_id : undefined,
  }));
  const label = (id: string) => options.find((option) => option.value === id)?.label ?? id;
  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">{t("customerPortal.hierarchy.memberTeamsHint")}</p>
      <div className={value.length > 1 ? "grid gap-4 sm:grid-cols-2" : "space-y-4"}>
        <div className="space-y-1.5">
          <Label htmlFor="member-teams">{t("customerPortal.hierarchy.teams")}</Label>
          <MultiSelect
            id="member-teams"
            options={options}
            value={[...value]}
            onValueChange={onChange}
            loading={teams.isLoading}
            disabled={!teams.isSuccess}
            placeholder={t("customerPortal.hierarchy.chooseTeams")}
          />
        </div>
        {value.length > 1 && (
          <div className="space-y-1.5">
            <Label htmlFor="member-default-team">{t("customerPortal.hierarchy.defaultTeam")}</Label>
            <Select
              value={value[0]}
              onValueChange={(id) => id && onChange([id, ...value.filter((team) => team !== id)])}
            >
              <SelectTrigger id="member-default-team" className="w-full">
                <SelectValue>{label(value[0])}</SelectValue>
              </SelectTrigger>
              <SelectContent>
                {value.map((id) => (
                  <SelectItem key={id} value={id}>
                    {label(id)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">{t("customerPortal.hierarchy.defaultTeamHint")}</p>
          </div>
        )}
      </div>
      {teams.isError && (
        <div role="alert" className="flex items-center gap-2 text-sm text-destructive">
          {t("customerPortal.hierarchy.teamsLoadError")}
          <Button type="button" variant="link" size="sm" onClick={() => void teams.refetch()}>
            {t("customerPortal.hierarchy.retryTeams")}
          </Button>
        </div>
      )}
    </div>
  );
}

function MoveForm({
  member,
  organizationId,
  onClose,
  pending,
  setPending,
}: {
  member: Member;
  organizationId: string;
  onClose: () => void;
  pending: boolean;
  setPending: (pending: boolean) => void;
}) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const { control, watch, getValues } = useZodForm(memberMoveSchema, {
    defaultValues: { teams: member.parent_team_ids },
  });
  const teams = watch("teams");
  const [error, setError] = useState<string | null>(null);
  const dirty = JSON.stringify(teams) !== JSON.stringify(member.parent_team_ids);
  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!teams.length || !dirty) return;
    const body = memberMoveSchema.parse(getValues());
    setPending(true);
    setError(null);
    try {
      await fetchClient.PATCH("/management/v1/customer_members/{customer_member_id}", {
        params: { path: { customer_member_id: member.customer_member_id } },
        body,
      });
      void queryClient.invalidateQueries({ queryKey: ["customer-hierarchy"] });
      void queryClient.invalidateQueries({ queryKey: ["get", "/management/v1/customer_portal/limits"] });
      toast.success(t("customerPortal.hierarchy.memberMoved"));
      onClose();
    } catch (cause) {
      setError(extractProxyErrorMessage(cause));
    } finally {
      setPending(false);
    }
  };
  return (
    <DialogContent className="sm:max-w-xl" showCloseButton={!pending}>
      <form className="space-y-5" onSubmit={(event) => void submit(event)}>
        <DialogHeader>
          <DialogTitle>{t("customerPortal.hierarchy.moveTeams")}</DialogTitle>
          <DialogDescription>{member.alias || member.customer_member_id}</DialogDescription>
        </DialogHeader>
        <fieldset disabled={pending}>
          <Controller
            control={control}
            name="teams"
            render={({ field }) => (
              <CustomerMemberTeams organizationId={organizationId} value={field.value} onChange={field.onChange} />
            )}
          />
        </fieldset>
        {!teams.length && (
          <p role="alert" className="text-sm text-destructive">
            {t("customerPortal.hierarchy.teamsRequired")}
          </p>
        )}
        {error && (
          <p role="alert" className="text-sm text-destructive">
            {error}
          </p>
        )}
        <DialogFooter>
          <Button type="button" variant="outline" disabled={pending} onClick={onClose}>
            {t("customerPortal.hierarchy.cancel")}
          </Button>
          <Button type="submit" disabled={pending || !dirty || !teams.length}>
            {t(pending ? "customerPortal.hierarchy.moving" : "customerPortal.hierarchy.moveTeams")}
          </Button>
        </DialogFooter>
      </form>
    </DialogContent>
  );
}

export default function MoveCustomerMemberDialog({
  member,
  organizationId,
  onClose,
}: {
  member: Member;
  organizationId: string;
  onClose: () => void;
}) {
  const [pending, setPending] = useState(false);
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open && !pending) onClose();
      }}
    >
      <MoveForm
        member={member}
        organizationId={organizationId}
        onClose={onClose}
        pending={pending}
        setPending={setPending}
      />
    </Dialog>
  );
}
