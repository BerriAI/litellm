import { useMemo, useState } from "react";
import { toast } from "sonner";
import {
  ArrowLeft,
  Copy,
  DollarSign,
  Edit,
  Key,
  Loader2,
  Users,
} from "lucide-react";
// eslint-disable-next-line litellm-ui/no-banned-ui-imports
import { BarChart } from "@tremor/react";
import { useProjectDetails } from "@/app/(dashboard)/hooks/projects/useProjectDetails";
import { useTeam } from "@/app/(dashboard)/hooks/teams/useTeams";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import DefaultProxyAdminTag from "../common_components/DefaultProxyAdminTag";
import { EditProjectModal } from "./ProjectModals/EditProjectModal";

interface TeamInfoShape {
  team_id: string;
  team_alias?: string;
  models?: string[];
  max_budget?: number | null;
  budget_duration?: string | null;
  spend?: number;
  members_with_roles?: { user_id: string; role: string }[];
}

interface ProjectDetailProps {
  projectId: string;
  onBack: () => void;
}

function EmptyState({ description }: { description: string }) {
  return (
    <div className="py-12 flex flex-col items-center justify-center text-muted-foreground">
      <div className="text-sm">{description}</div>
    </div>
  );
}

function CopyableId({ value }: { value: string }) {
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            onClick={() => {
              navigator.clipboard?.writeText(value);
              toast.success("Copied to clipboard");
            }}
            className="inline-flex items-center gap-1 text-xs font-mono rounded bg-muted px-1.5 py-0.5 hover:bg-muted/80"
          >
            {value}
            <Copy size={12} />
          </button>
        </TooltipTrigger>
        <TooltipContent>Copy</TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

export function ProjectDetail({ projectId, onBack }: ProjectDetailProps) {
  const { data: project, isLoading } = useProjectDetails(projectId);
  const { data: teamData } = useTeam(project?.team_id ?? undefined);
  // teamInfoCall returns { team_id, team_info: {...}, keys, team_memberships }
  const teamInfo: TeamInfoShape | undefined = ((teamData as unknown as {
    team_info?: TeamInfoShape;
  })?.team_info ?? teamData) as TeamInfoShape | undefined;
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
      <div className="p-6 md:px-12">
        <div className="flex justify-center items-center min-h-[300px]">
          <Loader2
            role="img"
            aria-hidden="true"
            className="h-10 w-10 animate-spin text-muted-foreground"
          />
        </div>
      </div>
    );
  }

  if (!project) {
    return (
      <div className="p-6 md:px-12">
        <Button variant="ghost" size="sm" onClick={onBack} className="mb-4">
          <ArrowLeft size={16} />
        </Button>
        <EmptyState description="Project not found" />
      </div>
    );
  }

  return (
    <main className="p-6 md:px-12">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="sm" onClick={onBack}>
            <ArrowLeft size={16} />
          </Button>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-2xl font-semibold m-0">
                {project.project_alias ?? project.project_id}
              </h2>
              <Badge variant={project.blocked ? "destructive" : "secondary"}>
                {project.blocked ? "Blocked" : "Active"}
              </Badge>
            </div>
            <div className="text-sm text-muted-foreground flex items-center gap-2">
              ID: <CopyableId value={project.project_id} />
            </div>
          </div>
        </div>
        <Button onClick={() => setIsEditModalVisible(true)}>
          <Edit size={16} />
          Edit Project
        </Button>
      </div>

      {/* Project Details */}
      <Card className="p-6 mb-6">
        <h3 className="text-lg font-semibold mb-3">Project Details</h3>
        <dl className="space-y-2 text-sm">
          <div className="flex gap-2">
            <dt className="w-36 text-muted-foreground">Description</dt>
            <dd>{project.description || "\u2014"}</dd>
          </div>
          <div className="flex gap-2">
            <dt className="w-36 text-muted-foreground">Created</dt>
            <dd>
              {new Date(project.created_at).toLocaleString()}
              {project.created_by && (
                <>
                  &nbsp;by&nbsp;
                  <DefaultProxyAdminTag userId={project.created_by} />
                </>
              )}
            </dd>
          </div>
          <div className="flex gap-2">
            <dt className="w-36 text-muted-foreground">Last Updated</dt>
            <dd>
              {new Date(project.updated_at).toLocaleString()}
              {project.updated_by && (
                <>
                  &nbsp;by&nbsp;
                  <DefaultProxyAdminTag userId={project.updated_by} />
                </>
              )}
            </dd>
          </div>
        </dl>
      </Card>

      {/* Spend / Budget */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
        <Card className="p-6 lg:col-span-1">
          <div className="flex items-center gap-2 mb-4 font-semibold">
            <DollarSign size={16} />
            Budget
          </div>
          <div className="space-y-4">
            <div>
              <span className="text-3xl font-semibold leading-none">
                ${spend.toFixed(2)}
              </span>
              <br />
              <span className="text-sm text-muted-foreground">
                {hasLimit
                  ? `of $${maxBudget.toFixed(2)} budget`
                  : "No budget limit"}
              </span>
            </div>
            {hasLimit && (
              <div>
                <Progress value={Math.round(spendPercent * 10) / 10} />
                <span className="text-xs text-muted-foreground">
                  {(Math.round(spendPercent * 10) / 10).toFixed(1)}% utilized
                </span>
              </div>
            )}
          </div>
        </Card>
        <Card className="p-6 lg:col-span-2">
          <div className="mb-4 font-semibold">Spend by Model</div>
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
            <EmptyState description="No model spend recorded yet" />
          )}
        </Card>
      </div>

      {/* Keys & Team */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
        <Card className="p-6">
          <div className="flex items-center gap-2 mb-4 font-semibold">
            <Key size={16} />
            Keys
          </div>
          <EmptyState description="No keys to display" />
        </Card>
        <Card className="p-6">
          <div className="flex items-center gap-2 mb-4 font-semibold">
            <Users size={16} />
            Team
          </div>
          {teamInfo ? (
            (() => {
              const teamBudget = teamInfo.max_budget ?? null;
              const teamSpend = teamInfo.spend ?? 0;
              const teamHasLimit = teamBudget != null && teamBudget > 0;
              const teamPercent = teamHasLimit
                ? Math.min((teamSpend / teamBudget) * 100, 100)
                : 0;

              return (
                <div className="flex flex-col gap-3">
                  {/* Team name + ID */}
                  <div>
                    <div className="font-semibold text-base">
                      {teamInfo.team_alias || teamInfo.team_id}
                    </div>
                    <div className="text-xs text-muted-foreground flex items-center gap-1">
                      ID: <CopyableId value={teamInfo.team_id} />
                    </div>
                  </div>

                  {/* Models */}
                  <div>
                    <div className="text-xs text-muted-foreground mb-1">
                      Models
                    </div>
                    {(teamInfo.models?.length ?? 0) > 0 ? (
                      <div className="flex flex-wrap gap-1 max-h-[60px] overflow-hidden">
                        {teamInfo.models?.map((m: string) => (
                          <Badge
                            key={m}
                            variant="secondary"
                            className="font-normal"
                          >
                            {m}
                          </Badge>
                        ))}
                      </div>
                    ) : (
                      <span className="text-sm text-muted-foreground">
                        All models
                      </span>
                    )}
                  </div>

                  {/* Budget + Spend compact */}
                  <div>
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-xs text-muted-foreground">
                        Spend
                      </span>
                      <span className="text-xs">
                        ${teamSpend.toFixed(2)}
                        {teamHasLimit ? (
                          <span className="text-muted-foreground">
                            {" "}
                            / ${teamBudget.toFixed(2)}
                          </span>
                        ) : (
                          <span className="text-muted-foreground">
                            {" "}
                            (Unlimited)
                          </span>
                        )}
                      </span>
                    </div>
                    {teamHasLimit && (
                      <Progress
                        value={Math.round(teamPercent * 10) / 10}
                        className="h-2"
                      />
                    )}
                  </div>

                  {/* Members */}
                  <div className="flex justify-between">
                    <span className="text-xs text-muted-foreground">
                      Members
                    </span>
                    <span className="text-xs">
                      {teamInfo.members_with_roles?.length ?? 0}
                    </span>
                  </div>
                </div>
              );
            })()
          ) : project.team_id ? (
            <div className="flex justify-center items-center py-4">
              <Skeleton className="h-6 w-32" />
            </div>
          ) : (
            <EmptyState description="No team assigned" />
          )}
        </Card>
      </div>

      {/* Edit Modal */}
      <EditProjectModal
        isOpen={isEditModalVisible}
        project={project}
        onClose={() => setIsEditModalVisible(false)}
      />
    </main>
  );
}
