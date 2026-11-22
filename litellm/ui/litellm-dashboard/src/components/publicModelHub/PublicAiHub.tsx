"use client";

import { useTranslation } from "react-i18next";

import { Tabs, TabsContent } from "@/components/ui/tabs";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ThemeProvider } from "@/contexts/ThemeContext";

import { HubHeader, type HubStatus } from "./HubHeader";
import { CatalogNavbar } from "./CatalogNavbar";
import { HubNotEnabled } from "./HubEmpty";
import { ModelsPane } from "./ModelsPane";
import { SkillsPane } from "./ResourcePanes";
import { type HubData, useHubData } from "./useHubData";
import { type HubTab, useHubUrlState } from "./useHubUrlState";
import { useRequirePublicHubAuth } from "./useRequirePublicHubAuth";
import { catalogTypography } from "./fonts";

const hubStatus = (hub: HubData): HubStatus => {
  if (hub.models.error && hub.skills.isError) return "unavailable";
  if (hub.models.isLoading || hub.skills.isPending) return "loading";
  return hub.models.error || hub.skills.isError ? "degraded" : "operational";
};

function PublicAiHubContent() {
  const { t } = useTranslation();
  useRequirePublicHubAuth();
  const hub = useHubData();
  const status = hubStatus(hub);
  const { tab, setTab, open, setOpen } = useHubUrlState();
  const paneProps = { selected: open, onSelect: setOpen };
  return (
    <Tabs
      value={tab}
      onValueChange={(next) => setTab(next as HubTab)}
      className={`${catalogTypography} min-h-dvh gap-0 bg-background`}
    >
      <CatalogNavbar
        title={hub.info.data?.docs_title || t("publicModelHub.header.title")}
        search={hub.models.searchValue}
        onSearchChange={(value) => {
          if (tab !== "models") setTab("models");
          hub.models.onSearchChange(value);
        }}
      />
      <main className="mx-auto w-full max-w-(--page-width) px-(--page-gutter) pb-12 max-[760px]:pb-7">
        <HubHeader info={hub.info.data} status={status} tab={tab} />
        {status === "unavailable" ? (
          <HubNotEnabled />
        ) : (
          <>
            <TabsContent value="models">
              <ModelsPane hub={hub} {...paneProps} />
            </TabsContent>
            <TabsContent value="skills">
              <SkillsPane
                rows={hub.skills.data}
                loading={hub.skills.isPending}
                error={hub.skills.isError}
                onRetry={() => void hub.skills.refetch()}
                {...paneProps}
              />
            </TabsContent>
          </>
        )}
      </main>
    </Tabs>
  );
}

export default function PublicAiHub({ accessToken }: { accessToken: string | null }) {
  return (
    <ThemeProvider accessToken={accessToken}>
      <TooltipProvider>
        <PublicAiHubContent />
      </TooltipProvider>
    </ThemeProvider>
  );
}
