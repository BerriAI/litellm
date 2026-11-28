"use client";

import { useState } from "react";
import { useWatch } from "react-hook-form";
import { useZodForm } from "@/lib/forms/useZodForm";
import { ShieldCheck } from "lucide-react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { toast } from "@/lib/toast";
import useAuthorized from "../../hooks/useAuthorized";
import { useOrganizations } from "../../hooks/organizations/useOrganizations";
import { AdminOnlyNotice } from "@/components/shared/AdminOnlyNotice";
import { ErrorState } from "@/components/shared/ErrorState";
import { PageContent } from "@/components/shared/Page";
import { PageHeader, PageHeaderTitle, PageHeaderDescription, PageHeaderControls } from "@/components/shared/PageHeader";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import OrganizationDropdown from "@/components/common_components/OrganizationDropdown";
import TeamDropdown from "@/components/common_components/team_dropdown";
import UserDropdown from "@/components/common_components/UserDropdown";
import AppAccessSelector from "./AppAccessSelector";
import { appAccessChanged, GRANTEE_KINDS, loadGranteeApps, saveGranteeApps, type GranteeKind } from "./appAccessApi";
import { initialInternalAccessValues } from "./internalAccessMapper";
import { internalAccessSchema } from "./internalAccessSchema";
import { grantedAppsKey, readsEveryApp } from "./useGrantedApps";

interface Grantee {
  kind: GranteeKind;
  id: string;
}

function GranteeAppAccess({ grantee, accessToken }: { grantee: Grantee; accessToken: string }) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const queryKey = ["apps", "internal-access", grantee.kind, grantee.id] as const;
  const stored = useQuery({ queryKey, queryFn: () => loadGranteeApps(grantee.kind, grantee.id, accessToken) });
  const form = useZodForm(internalAccessSchema, { defaultValues: initialInternalAccessValues });
  const selected = useWatch({ control: form.control, name: "draft" });
  const value = selected ?? stored.data ?? [];
  const save = useMutation({
    mutationFn: () => saveGranteeApps(grantee.kind, grantee.id, value, accessToken),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey });
      await queryClient.invalidateQueries({ queryKey: grantedAppsKey });
      form.setValue("draft", null);
      toast.success(t("apps.internalAccess.saved"));
    },
  });
  if (stored.isError) return <ErrorState onRetry={() => void stored.refetch()} />;
  if (stored.isPending) return <p>{t("customerPortal.common.loading")}</p>;
  return (
    <div className="space-y-3">
      <label htmlFor="internal-access-apps" className="text-sm font-medium">
        {t("apps.internalAccess.apps")}
      </label>
      <AppAccessSelector
        id="internal-access-apps"
        value={value}
        onChange={(apps) => form.setValue("draft", apps)}
        placeholder={t("apps.internalAccess.appsPlaceholder")}
      />
      {save.isError && (
        <p role="alert" className="text-sm text-destructive">
          {t("apps.internalAccess.saveFailed")}
        </p>
      )}
      <Button disabled={save.isPending || !appAccessChanged(stored.data, value)} onClick={() => save.mutate()}>
        {t("apps.internalAccess.save")}
      </Button>
    </div>
  );
}

function GranteePicker({
  grantee,
  onPick,
}: {
  grantee: Grantee | { kind: GranteeKind; id: null };
  onPick: (id: string | null) => void;
}) {
  const organizations = useOrganizations();
  if (grantee.kind === "team")
    return <TeamDropdown id="internal-access-grantee" value={grantee.id} onChange={onPick} />;
  if (grantee.kind === "user")
    return <UserDropdown id="internal-access-grantee" value={grantee.id} onChange={onPick} />;
  return (
    <OrganizationDropdown
      id="internal-access-grantee"
      organizations={organizations.data}
      loading={organizations.isLoading}
      value={grantee.id}
      onChange={onPick}
    />
  );
}

export default function InternalAccess() {
  const { t } = useTranslation();
  const { accessToken, userRole, isViewOnly } = useAuthorized();
  const [kind, setKind] = useState<GranteeKind>("team");
  const [id, setId] = useState<string | null>(null);
  if (!accessToken || !readsEveryApp(userRole, isViewOnly))
    return <AdminOnlyNotice framed pageTitle={t("apps.internalAccess.title")} />;
  const pickKind = (next: GranteeKind) => {
    setKind(next);
    setId(null);
  };
  return (
    <>
      <PageHeader>
        <PageHeaderTitle>
          <ShieldCheck />
          {t("apps.internalAccess.title")}
        </PageHeaderTitle>
        <PageHeaderDescription>{t("apps.internalAccess.hint")}</PageHeaderDescription>
        <PageHeaderControls>
          <Tabs
            value={kind}
            onValueChange={(next) => pickKind(GRANTEE_KINDS.find((known) => known === next) ?? "team")}
          >
            <TabsList>
              {GRANTEE_KINDS.map((known) => (
                <TabsTrigger key={known} value={known}>
                  {t(`apps.internalAccess.kinds.${known}`)}
                </TabsTrigger>
              ))}
            </TabsList>
          </Tabs>
        </PageHeaderControls>
      </PageHeader>
      <PageContent className="max-w-2xl">
        <div className="space-y-3">
          <label htmlFor="internal-access-grantee" className="text-sm font-medium">
            {t(`apps.internalAccess.kinds.${kind}`)}
          </label>
          <GranteePicker grantee={{ kind, id }} onPick={setId} />
        </div>
        {id && <GranteeAppAccess key={`${kind}:${id}`} grantee={{ kind, id }} accessToken={accessToken} />}
      </PageContent>
    </>
  );
}
