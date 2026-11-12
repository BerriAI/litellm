import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { downloadBlob } from "@/components/EntityUsageExport/utils";
import { apiClient } from "@/components/networking";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { ApiError } from "@/lib/http/client";
import type { components } from "@/lib/http/schema";

type Versions = components["schemas"]["SkillHubVersions"];
type PublishedVersion = components["schemas"]["SkillHubVersion"];
type Skill = components["schemas"]["SkillHubSkill"];
type VersionedSkill = Pick<Skill, "id" | "name" | "version" | "updated_at">;

function VersionArchive({
  skillName,
  version,
  accessToken,
}: {
  skillName: string;
  version: PublishedVersion;
  accessToken: string;
}) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [error, setError] = useState<string | null>(null);
  const [downloading, setDownloading] = useState(false);
  const download = async () => {
    if (!version.download_url || downloading) return;
    setDownloading(true);
    setError(null);
    try {
      const content = await apiClient.getBlob(
        `/skill-hub/skills/${encodeURIComponent(skillName)}/versions/${encodeURIComponent(version.id)}/archive`,
        { accessToken, redirect: "error" },
      );
      downloadBlob(content, `skill-${version.version ?? "legacy"}.zip`);
      await queryClient.invalidateQueries({ queryKey: ["skill-hub", accessToken] });
    } catch {
      setError(t("claudeCodePlugins.skillDetail.history.downloadFailed"));
    } finally {
      setDownloading(false);
    }
  };
  return (
    <>
      <Button variant="outline" size="sm" onClick={download} disabled={downloading}>
        {t("claudeCodePlugins.skillDetail.history.download")}
      </Button>
      {error && <p role="alert">{error}</p>}
    </>
  );
}

function RestoreVersion({
  skill,
  version,
  accessToken,
  onRestored,
}: {
  skill: VersionedSkill;
  version: PublishedVersion;
  accessToken: string;
  onRestored: (skill: Skill) => void;
}) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const [newVersion, setNewVersion] = useState("");
  const [restoring, setRestoring] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const restore = async () => {
    if (restoring) return;
    setRestoring(true);
    setError(null);
    try {
      const restored = await apiClient.post<Skill>(
        `/skill-hub/skills/${encodeURIComponent(skill.name)}/versions/${encodeURIComponent(version.id)}/restore`,
        {
          body: {
            expected_skill_id: skill.id,
            expected_updated_at: skill.updated_at ?? null,
            version: newVersion.trim() || null,
          } satisfies components["schemas"]["SkillHubRestore"],
          accessToken,
        },
      );
      setOpen(false);
      onRestored(restored);
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["skill-hub", accessToken], refetchType: "all" }),
        queryClient.invalidateQueries({ queryKey: ["skill-versions", skill.id], refetchType: "all" }),
      ]);
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : t("claudeCodePlugins.skillDetail.history.restoreFailed"));
    } finally {
      setRestoring(false);
    }
  };
  return (
    <>
      <Button variant="outline" size="sm" onClick={() => setOpen(true)}>
        {t("claudeCodePlugins.skillDetail.history.restore")}
      </Button>
      <Dialog open={open} onOpenChange={(next) => !restoring && setOpen(next)}>
        <DialogContent showCloseButton={!restoring}>
          <DialogHeader>
            <DialogTitle>
              {t("claudeCodePlugins.skillDetail.history.restoreTitle", { version: version.version })}
            </DialogTitle>
            <DialogDescription>{t("claudeCodePlugins.skillDetail.history.restoreDescription")}</DialogDescription>
          </DialogHeader>
          <form
            className="space-y-4"
            onSubmit={(event) => {
              event.preventDefault();
              void restore();
            }}
          >
            <label className="block space-y-2">
              <span>{t("claudeCodePlugins.skillDetail.history.newVersion")}</span>
              <Input value={newVersion} onChange={(event) => setNewVersion(event.target.value)} disabled={restoring} />
            </label>
            <p className="text-sm text-muted-foreground">{t("claudeCodePlugins.skillDetail.history.versionHint")}</p>
            {error && <p role="alert">{error}</p>}
            <Button type="submit" disabled={restoring}>
              {t(
                restoring
                  ? "claudeCodePlugins.skillDetail.history.restoring"
                  : "claudeCodePlugins.skillDetail.history.confirmRestore",
              )}
            </Button>
          </form>
        </DialogContent>
      </Dialog>
    </>
  );
}

export default function SkillVersionHistory({
  skill,
  accessToken,
  isAdmin = false,
  onRestored,
}: {
  skill: VersionedSkill;
  accessToken: string;
  isAdmin?: boolean;
  onRestored?: (skill: Skill) => void;
}) {
  const { t } = useTranslation();
  const [offset, setOffset] = useState(0);
  const onAdminRestore = isAdmin ? onRestored : undefined;
  const history = useQuery({
    queryKey: ["skill-versions", skill.id, skill.updated_at, accessToken, offset],
    queryFn: () =>
      apiClient.get<Versions>(`/skill-hub/skills/${encodeURIComponent(skill.name)}/versions`, {
        accessToken,
        query: { limit: 20, offset },
      }),
  });
  return (
    <section className="space-y-4" aria-label={t("claudeCodePlugins.skillDetail.history.title")}>
      <p className="text-sm text-muted-foreground">{t("claudeCodePlugins.skillDetail.history.explanation")}</p>
      {history.isPending && <p>{t("claudeCodePlugins.skillDetail.history.loading")}</p>}
      {history.isError && <p role="alert">{t("claudeCodePlugins.skillDetail.history.failed")}</p>}
      {history.data?.versions.length === 0 && <p>{t("claudeCodePlugins.skillDetail.history.empty")}</p>}
      {history.data?.versions.map((version) => (
        <article key={version.id} className="rounded-lg border border-border p-4">
          <div className="flex items-center justify-between gap-4">
            <h3 className="font-medium">{version.version ?? t("claudeCodePlugins.skillDetail.history.legacy")}</h3>
            {version.published_at && (
              <time className="text-sm text-muted-foreground">{new Date(version.published_at).toLocaleString()}</time>
            )}
          </div>
          {version.description && <p className="mt-2 text-sm">{version.description}</p>}
          <p className="my-2 text-sm text-muted-foreground">
            {t(
              version.content_kind === "stored_archive"
                ? "claudeCodePlugins.skillDetail.history.retained"
                : "claudeCodePlugins.skillDetail.history.reference",
            )}
          </p>
          {version.content_kind === "stored_archive" && (
            <div className="flex gap-2">
              <VersionArchive skillName={skill.name} version={version} accessToken={accessToken} />
              {onAdminRestore && version.version !== skill.version && (
                <RestoreVersion
                  key={`${skill.id}:${skill.updated_at}:${accessToken}`}
                  skill={skill}
                  version={version}
                  accessToken={accessToken}
                  onRestored={onAdminRestore}
                />
              )}
            </div>
          )}
          <details className="mt-3 text-sm">
            <summary className="cursor-pointer">{t("claudeCodePlugins.skillDetail.history.metadata")}</summary>
            <pre className="mt-2 overflow-auto whitespace-pre-wrap break-all rounded bg-muted p-3">
              {JSON.stringify(version.manifest, null, 2)}
            </pre>
          </details>
        </article>
      ))}
      <div className="flex gap-2">
        <Button variant="outline" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 20))}>
          {t("claudeCodePlugins.skillDetail.history.newer")}
        </Button>
        <Button
          variant="outline"
          disabled={history.data?.next_offset == null}
          onClick={() => setOffset(history.data?.next_offset ?? offset)}
        >
          {t("claudeCodePlugins.skillDetail.history.older")}
        </Button>
      </div>
    </section>
  );
}
