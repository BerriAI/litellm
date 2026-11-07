"use client";

import localFont from "next/font/local";
import { useTranslation } from "react-i18next";

import Navbar from "@/components/navbar";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ThemeProvider } from "@/contexts/ThemeContext";

import { HubHeader, type HubStatus } from "./HubHeader";
import { HubNotEnabled } from "./HubEmpty";
import { ModelsPane } from "./ModelsPane";
import { SkillsPane } from "./ResourcePanes";
import { type HubData, useHubData } from "./useHubData";
import { type HubTab, useHubUrlState } from "./useHubUrlState";
import { useRequirePublicHubAuth } from "./useRequirePublicHubAuth";
import styles from "./catalog.module.css";

const displayFont = localFont({
  src: "./fonts/barlow-condensed-latin-600.ttf",
  weight: "600",
  variable: "--font-catalog-display",
});

const hubStatus = (hub: HubData): HubStatus => {
  if (hub.models.error && hub.skills.isError) return "unavailable";
  if (hub.models.isLoading || hub.skills.isPending) return "loading";
  return hub.models.error || hub.skills.isError ? "degraded" : "operational";
};

function PublicAiHubContent({ accessToken }: { accessToken: string | null }) {
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
      className={`${styles.catalog} ${displayFont.variable}`}
    >
      <Navbar
        accessToken={accessToken}
        isPublicPage
        publicNavigation={
          <TabsList variant="line" className={styles.navigation} aria-label={t("publicModelHub.header.title")}>
            <TabsTrigger value="models">{t("publicModelHub.catalog.modelHub")}</TabsTrigger>
            <TabsTrigger value="skills">{t("publicModelHub.catalog.skillsHub")}</TabsTrigger>
          </TabsList>
        }
      />
      <main className={styles.main}>
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
        <PublicAiHubContent accessToken={accessToken} />
      </TooltipProvider>
    </ThemeProvider>
  );
}
