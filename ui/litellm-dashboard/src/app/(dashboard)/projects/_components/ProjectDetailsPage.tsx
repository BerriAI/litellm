import { useCanManageProjects } from "@/app/(dashboard)/hooks/projects/projectAccess";
import { useProjectDetails } from "@/app/(dashboard)/hooks/projects/useProjectDetails";
import { useTeam } from "@/app/(dashboard)/hooks/teams/useTeams";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { isUserTeamAdminForSingleTeam } from "@/utils/roles";
import { BarChart } from "@/components/shared/charts";
import { ArrowLeftIcon, DollarSignIcon, EditIcon, UsersIcon } from "lucide-react";
import { useMemo, useState } from "react";
import { UserReference } from "@/components/shared/EntityReference";
import CopyButton from "@/components/shared/CopyButton";
import { useUserDisplayNames } from "@/app/(dashboard)/hooks/users/useUsers";
import { StatusBadge } from "@/components/shared/table_cells/status_badge";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Meter, MeterIndicator, MeterTrack } from "@/components/shared/Meter";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { EditProjectModal } from "./ProjectModals/EditProjectModal";
import { ProjectKeysSection } from "./ProjectKeysSection";

interface ProjectDetailProps {
  projectId: string;
  onBack: () => void;
}

const utilisationTone = (percent: number) => (percent >= 90 ? "over" : percent >= 70 ? "warning" : "default");

export function ProjectDetail({ projectId, onBack }: ProjectDetailProps) {
  const { data: project, isLoading } = useProjectDetails(projectId);
  const { data: teamInfo } = useTeam(project?.team_id ?? undefined);
  const { userId } = useAuthorized();
  const { data: displayNames } = useUserDisplayNames(
    useMemo(() => [project?.created_by ?? "", project?.updated_by ?? ""], [project]),
  );
  const canEditProject = useCanManageProjects(
    isUserTeamAdminForSingleTeam(teamInfo?.members_with_roles ?? null, userId ?? ""),
  );
  const [isEditModalVisible, setIsEditModalVisible] = useState(false);

  const spend = project?.spend ?? 0;
  const maxBudget = project?.litellm_budget_table?.max_budget ?? null;
  const hasLimit = maxBudget != null && maxBudget > 0;
  const spendPercent = hasLimit ? Math.min((spend / maxBudget) * 100, 100) : 0;

  const modelSpendData = useMemo(() => {
    const raw = (project?.model_spend ?? {}) as Record<string, number>;
    return Object.entries(raw)
      .map(([model, value]) => ({ model, spend: value }))
      .sort((a, b) => b.spend - a.spend);
  }, [project?.model_spend]);

  if (isLoading) {
    return (
      <div className="p-6 px-12">
        <div
          role="status"
          aria-busy="true"
          aria-label="Loading"
          className="flex min-h-[300px] items-center justify-center"
        >
          <UiLoadingSpinner className="size-8 text-primary" />
        </div>
      </div>
    );
  }

  if (!project) {
    return (
      <div className="p-6 px-12">
        <Button variant="ghost" size="icon" aria-label="Back" onClick={onBack} className="mb-4">
          <ArrowLeftIcon className="size-4" />
        </Button>
        <p className="py-8 text-center text-sm text-muted-foreground">Project not found</p>
      </div>
    );
  }

  return (
    <div className="p-6 px-12">
      <div className="mb-6 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="icon" aria-label="Back" onClick={onBack}>
            <ArrowLeftIcon className="size-4" />
          </Button>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-semibold tracking-tight text-foreground">
                {project.project_alias ?? project.project_id}
              </h1>
              <StatusBadge
                tone={project.blocked ? "error" : "success"}
                label={project.blocked ? "Blocked" : "Active"}
              />
            </div>
            <div className="flex items-center gap-1 text-sm text-muted-foreground">
              <span>ID: {project.project_id}</span>
              <CopyButton value={project.project_id} label="Copy project ID" />
            </div>
          </div>
        </div>
        {canEditProject && (
          <Button onClick={() => setIsEditModalVisible(true)}>
            <EditIcon className="size-4" />
            Edit Project
          </Button>
        )}
      </div>

      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Project Details</CardTitle>
        </CardHeader>
        <CardContent>
          <dl className="grid w-1/2 grid-cols-[max-content_minmax(0,1fr)] gap-x-4 gap-y-2 text-sm">
            <dt className="text-muted-foreground">Description</dt>
            <dd className="text-foreground">{project.description || "—"}</dd>
            <dt className="text-muted-foreground">Created</dt>
            <dd className="flex min-w-0 flex-wrap items-center gap-x-1 text-foreground">
              <span className="shrink-0 whitespace-nowrap">{new Date(project.created_at).toLocaleString()}</span>
              {project.created_by && (
                <span className="flex min-w-0 grow basis-24 items-center gap-1">
                  <span className="shrink-0">by</span>
                  <UserReference userId={project.created_by} displayName={displayNames?.[project.created_by]} />
                </span>
              )}
            </dd>
            <dt className="text-muted-foreground">Last Updated</dt>
            <dd className="flex min-w-0 flex-wrap items-center gap-x-1 text-foreground">
              <span className="shrink-0 whitespace-nowrap">{new Date(project.updated_at).toLocaleString()}</span>
              {project.updated_by && (
                <span className="flex min-w-0 grow basis-24 items-center gap-1">
                  <span className="shrink-0">by</span>
                  <UserReference userId={project.updated_by} displayName={displayNames?.[project.updated_by]} />
                </span>
              )}
            </dd>
          </dl>
        </CardContent>
      </Card>

      <div className="mb-6 grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Card className="h-full">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <DollarSignIcon className="size-4" />
              Budget
            </CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <div>
              <p className="text-[28px] leading-none font-medium text-foreground">${spend.toFixed(2)}</p>
              <p className="mt-1 text-sm text-muted-foreground">
                {hasLimit ? `of $${maxBudget.toFixed(2)} budget` : "No budget limit"}
              </p>
            </div>
            {hasLimit && (
              <div>
                <Meter value={Math.round(spendPercent * 10) / 10}>
                  <MeterTrack>
                    <MeterIndicator tone={utilisationTone(spendPercent)} />
                  </MeterTrack>
                </Meter>
                <p className="mt-1 text-xs text-muted-foreground">
                  {(Math.round(spendPercent * 10) / 10).toFixed(1)}% utilized
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="h-full lg:col-span-2">
          <CardHeader>
            <CardTitle>Spend by Model</CardTitle>
          </CardHeader>
          <CardContent>
            {modelSpendData.length > 0 ? (
              <BarChart
                data={modelSpendData}
                index="model"
                categories={["spend"]}
                colors={["cyan"]}
                layout="vertical"
                valueFormatter={(value) => `$${value.toFixed(4)}`}
                yAxisWidth={140}
                showLegend={false}
                style={{ height: Math.max(modelSpendData.length * 40, 120) }}
              />
            ) : (
              <p className="py-8 text-center text-sm text-muted-foreground">No model spend recorded yet</p>
            )}
          </CardContent>
        </Card>
      </div>

      <div className="mb-6 grid grid-cols-1 gap-4 lg:grid-cols-2">
        <ProjectKeysSection projectId={projectId} />

        <Card className="h-full">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <UsersIcon className="size-4" />
              Team
            </CardTitle>
          </CardHeader>
          <CardContent>
            {teamInfo ? (
              (() => {
                const teamBudget = teamInfo.max_budget ?? null;
                const teamSpend = teamInfo.spend ?? 0;
                const teamHasLimit = teamBudget != null && teamBudget > 0;
                const teamPercent = teamHasLimit ? Math.min((teamSpend / teamBudget) * 100, 100) : 0;

                return (
                  <div className="flex flex-col gap-3">
                    <div>
                      <p className="text-base font-medium text-foreground">{teamInfo.team_alias || teamInfo.team_id}</p>
                      <div className="flex items-center gap-1 text-xs text-muted-foreground">
                        <span>ID: {teamInfo.team_id}</span>
                        <CopyButton value={teamInfo.team_id} label="Copy team ID" />
                      </div>
                    </div>

                    <div>
                      <p className="mb-1 text-xs text-muted-foreground">Models</p>
                      {(teamInfo.models?.length ?? 0) > 0 ? (
                        <div className="flex max-h-[60px] flex-wrap gap-1 overflow-hidden">
                          {teamInfo.models?.map((m: string) => (
                            <Badge key={m} variant="outline">
                              {m}
                            </Badge>
                          ))}
                        </div>
                      ) : (
                        <p className="text-sm text-muted-foreground">All models</p>
                      )}
                    </div>

                    <div>
                      <div className="mb-0.5 flex items-center justify-between">
                        <span className="text-xs text-muted-foreground">Spend</span>
                        <span className="text-xs text-foreground">
                          ${teamSpend.toFixed(2)}
                          <span className="text-muted-foreground">
                            {teamHasLimit ? ` / $${teamBudget.toFixed(2)}` : " (Unlimited)"}
                          </span>
                        </span>
                      </div>
                      {teamHasLimit && (
                        <Meter value={Math.round(teamPercent * 10) / 10}>
                          <MeterTrack>
                            <MeterIndicator tone={utilisationTone(teamPercent)} />
                          </MeterTrack>
                        </Meter>
                      )}
                    </div>

                    <div className="flex items-center justify-between">
                      <span className="text-xs text-muted-foreground">Members</span>
                      <span className="text-xs text-foreground">{teamInfo.members_with_roles?.length ?? 0}</span>
                    </div>
                  </div>
                );
              })()
            ) : project.team_id ? (
              <div
                role="status"
                aria-busy="true"
                aria-label="Loading team"
                className="flex items-center justify-center p-4"
              >
                <UiLoadingSpinner className="size-5 text-muted-foreground" />
              </div>
            ) : (
              <p className="py-8 text-center text-sm text-muted-foreground">No team assigned</p>
            )}
          </CardContent>
        </Card>
      </div>

      <EditProjectModal isOpen={isEditModalVisible} project={project} onClose={() => setIsEditModalVisible(false)} />
    </div>
  );
}
