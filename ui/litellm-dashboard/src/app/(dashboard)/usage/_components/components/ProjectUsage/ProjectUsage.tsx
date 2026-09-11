import React, { useMemo, useState } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { Info } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { BarChart } from "@/components/shared/charts";
import { ChartLoader } from "@/components/shared/chart_loader";
import { MultiSelect, type MultiSelectOption } from "@/components/shared/MultiSelect";
import type { DateRangePickerValue } from "@/components/shared/date_picker_types";
import { Card as ShadcnCard, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { projectDailyActivityCall } from "@/components/networking";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { valueFormatterSpend } from "@/components/UsagePage/utils/value_formatters";

import { buildSummaryTiles } from "../EntityUsage/entityUsageSummary";
import { SummaryTileCard } from "../EntityUsage/SummaryTileCard";
import type { EntityList } from "../EntityUsage/EntityUsage";
import ProjectSpendBreakdown from "./ProjectSpendBreakdown";
import { buildDailySpendSeries, buildProjectSpendBreakdown, summarizeProjectUsage } from "./projectUsageAggregations";

interface ProjectUsageProps {
  accessToken: string | null;
  projectList: EntityList[] | null;
  dateValue: DateRangePickerValue;
  premiumUser: boolean;
}

const ProjectUsage: React.FC<ProjectUsageProps> = ({ accessToken, projectList, dateValue, premiumUser }) => {
  const [selectedProjectIds, setSelectedProjectIds] = useState<string[]>([]);

  const startTime = useMemo(() => (dateValue.from ? new Date(dateValue.from) : null), [dateValue.from]);
  const endTime = useMemo(() => (dateValue.to ? new Date(dateValue.to) : null), [dateValue.to]);

  const projectOptions = useMemo<MultiSelectOption[]>(
    () => (projectList ?? []).map((project) => ({ label: project.label || project.value, value: project.value })),
    [projectList],
  );

  const hasSelection = selectedProjectIds.length > 0;
  const hasDateRange = !!startTime && !!endTime;
  const hasRequestWindow = premiumUser && !!accessToken && hasDateRange;
  const enabled = hasRequestWindow && hasSelection;

  const queryOptions = {
    queryKey: ["project-daily-activity", selectedProjectIds, startTime?.toISOString(), endTime?.toISOString()],
    queryFn: () =>
      projectDailyActivityCall(accessToken as string, startTime as Date, endTime as Date, selectedProjectIds),
    enabled,
    placeholderData: keepPreviousData,
  };
  const { data, isPending, isFetching, isPlaceholderData, isError, error } = useQuery(queryOptions);

  const rows = useMemo(() => data?.results ?? [], [data]);
  const summary = useMemo(() => summarizeProjectUsage(rows), [rows]);
  const dailySpend = useMemo(() => buildDailySpendSeries(rows), [rows]);
  const projectBreakdown = useMemo(() => buildProjectSpendBreakdown(rows), [rows]);

  if (!premiumUser) {
    return (
      <Alert variant="info">
        <AlertTitle>Project Usage is an Enterprise feature</AlertTitle>
        <AlertDescription>
          Filtering usage by project requires a LiteLLM Enterprise license. Get a 7 day trial at{" "}
          <a href="https://www.litellm.ai/enterprise#trial" target="_blank" rel="noreferrer" className="underline">
            litellm.ai/enterprise
          </a>
          .
        </AlertDescription>
      </Alert>
    );
  }

  const isLoadingRows = isPending || (isFetching && isPlaceholderData);

  const renderResultsPanel = () => {
    if (!hasSelection) {
      return (
        <div className="col-span-2">
          <ShadcnCard>
            <CardContent>
              <p className="text-sm text-muted-foreground py-8 text-center">
                Select at least one project above to view its usage.
              </p>
            </CardContent>
          </ShadcnCard>
        </div>
      );
    }

    if (isError) {
      return (
        <div className="col-span-2">
          <Alert variant="error">
            <AlertTitle>Could not load project usage</AlertTitle>
            <AlertDescription>{extractProxyErrorMessage(error)}</AlertDescription>
          </Alert>
        </div>
      );
    }

    return (
      <>
        <div className="col-span-2">
          <ShadcnCard>
            <CardContent>
              <h3 className="text-lg font-medium text-foreground">Project Spend Overview</h3>
              {isLoadingRows ? (
                <ChartLoader isDateChanging={false} />
              ) : (
                <div className="grid grid-cols-5 gap-4 mt-4">
                  {buildSummaryTiles(summary, false).map((tile) => (
                    <SummaryTileCard key={tile.title} tile={tile} />
                  ))}
                </div>
              )}
            </CardContent>
          </ShadcnCard>
        </div>

        <div className="col-span-2">
          <ShadcnCard>
            <CardHeader>
              <CardTitle className="text-base font-semibold">Daily Spend</CardTitle>
            </CardHeader>
            <CardContent>
              {isLoadingRows ? (
                <ChartLoader isDateChanging={false} />
              ) : (
                <BarChart
                  data={dailySpend}
                  index="date"
                  categories={["spend"]}
                  colors={["cyan"]}
                  valueFormatter={valueFormatterSpend}
                  yAxisWidth={100}
                />
              )}
            </CardContent>
          </ShadcnCard>
        </div>

        <div className="col-span-2">
          <ProjectSpendBreakdown loading={isLoadingRows} isDateChanging={false} projectSpend={projectBreakdown} />
        </div>
      </>
    );
  };

  return (
    <div className="grid grid-cols-2 gap-2 w-full">
      <div className="col-span-2">
        <ShadcnCard>
          <CardContent>
            <div className="flex items-center gap-2 mb-2">
              <h3 className="text-sm font-medium text-foreground">Projects</h3>
              <Tooltip>
                <TooltipTrigger render={<Info className="size-4 text-muted-foreground hover:text-foreground" />} />
                <TooltipContent>Select one or more projects to compare their usage</TooltipContent>
              </Tooltip>
            </div>
            <MultiSelect
              options={projectOptions}
              value={selectedProjectIds}
              onValueChange={setSelectedProjectIds}
              placeholder="Search or select projects..."
              emptyText="No projects found"
            />
          </CardContent>
        </ShadcnCard>
      </div>

      {renderResultsPanel()}
    </div>
  );
};

export default ProjectUsage;
