import React, { useEffect, useMemo, useState } from "react";
import { X } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Tooltip as ShTooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

// eslint-disable-next-line litellm-ui/no-banned-ui-imports
import { BarChart, DateRangePickerValue } from "@tremor/react";

import {
  userAgentSummaryCall,
  tagDauCall,
  tagWauCall,
  tagMauCall,
  tagDistinctCall,
} from "./networking";
import PerUserUsage from "./per_user_usage";
import { ChartLoader } from "./shared/chart_loader";

interface TagActiveUsersResponse {
  tag: string;
  active_users: number;
  date: string;
  period_start?: string;
  period_end?: string;
}

interface ActiveUsersAnalyticsResponse {
  results: TagActiveUsersResponse[];
}

interface TagSummaryMetrics {
  tag: string;
  unique_users: number;
  total_requests: number;
  successful_requests: number;
  failed_requests: number;
  total_tokens: number;
  total_spend: number;
}

interface TagSummaryResponse {
  results: TagSummaryMetrics[];
}

interface DistinctTagResponse {
  tag: string;
}

interface UserAgentActivityProps {
  accessToken: string | null;
  userRole: string | null;
  dateValue: DateRangePickerValue;
  onDateChange?: (value: DateRangePickerValue) => void;
}

/**
 * shadcn Select + badge chip multi-select for tag filtering.
 */
function TagMultiSelect({
  value,
  onChange,
  options,
  placeholder,
}: {
  value: string[];
  onChange: (next: string[]) => void;
  options: { label: string; value: string; title?: string }[];
  placeholder: string;
}) {
  const selected = useMemo(() => value ?? [], [value]);
  const remaining = useMemo(
    () => options.filter((o) => !selected.includes(o.value)),
    [options, selected],
  );

  return (
    <div className="space-y-2">
      <Select
        value=""
        onValueChange={(v) => {
          if (v) onChange([...selected, v]);
        }}
      >
        <SelectTrigger>
          <SelectValue placeholder={placeholder} />
        </SelectTrigger>
        <SelectContent>
          {remaining.length === 0 ? (
            <div className="py-2 px-3 text-sm text-muted-foreground">
              No more options
            </div>
          ) : (
            remaining.map((opt) => (
              <SelectItem key={opt.value} value={opt.value} title={opt.title}>
                {opt.label}
              </SelectItem>
            ))
          )}
        </SelectContent>
      </Select>
      {selected.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {selected.map((v) => {
            const opt = options.find((o) => o.value === v);
            return (
              <Badge
                key={v}
                variant="secondary"
                className="flex items-center gap-1 max-w-[240px]"
              >
                <span className="truncate">{opt?.label ?? v}</span>
                <button
                  type="button"
                  onClick={() => onChange(selected.filter((s) => s !== v))}
                  className="inline-flex items-center justify-center rounded-full hover:bg-muted-foreground/20"
                  aria-label={`Remove ${opt?.label ?? v}`}
                >
                  <X size={12} />
                </button>
              </Badge>
            );
          })}
        </div>
      )}
    </div>
  );
}

const UserAgentActivity: React.FC<UserAgentActivityProps> = ({
  accessToken,
  userRole,
  dateValue,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  onDateChange,
}) => {
  const MAX_CATEGORIES = 10;

  const [dauData, setDauData] = useState<ActiveUsersAnalyticsResponse>({
    results: [],
  });
  const [wauData, setWauData] = useState<ActiveUsersAnalyticsResponse>({
    results: [],
  });
  const [mauData, setMauData] = useState<ActiveUsersAnalyticsResponse>({
    results: [],
  });
  const [summaryData, setSummaryData] = useState<TagSummaryResponse>({
    results: [],
  });

  const [userAgentFilter] = useState<string>("");

  const [availableTags, setAvailableTags] = useState<string[]>([]);
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [tagsLoading, setTagsLoading] = useState(false);

  const [dauLoading, setDauLoading] = useState(false);
  const [wauLoading, setWauLoading] = useState(false);
  const [mauLoading, setMauLoading] = useState(false);
  const [summaryLoading, setSummaryLoading] = useState(false);

  const today = new Date();

  const fetchAvailableTags = async () => {
    if (!accessToken) return;

    setTagsLoading(true);
    try {
      const data = await tagDistinctCall(accessToken);
      setAvailableTags(data.results.map((item: DistinctTagResponse) => item.tag));
    } catch (error) {
      console.error("Failed to fetch available tags:", error);
    } finally {
      setTagsLoading(false);
    }
  };

  const fetchDauData = async () => {
    if (!accessToken) return;

    setDauLoading(true);
    try {
      const data = await tagDauCall(
        accessToken,
        today,
        userAgentFilter || undefined,
        selectedTags.length > 0 ? selectedTags : undefined,
      );
      setDauData(data);
    } catch (error) {
      console.error("Failed to fetch DAU data:", error);
    } finally {
      setDauLoading(false);
    }
  };

  const fetchWauData = async () => {
    if (!accessToken) return;

    setWauLoading(true);
    try {
      const data = await tagWauCall(
        accessToken,
        today,
        userAgentFilter || undefined,
        selectedTags.length > 0 ? selectedTags : undefined,
      );
      setWauData(data);
    } catch (error) {
      console.error("Failed to fetch WAU data:", error);
    } finally {
      setWauLoading(false);
    }
  };

  const fetchMauData = async () => {
    if (!accessToken) return;

    setMauLoading(true);
    try {
      const data = await tagMauCall(
        accessToken,
        today,
        userAgentFilter || undefined,
        selectedTags.length > 0 ? selectedTags : undefined,
      );
      setMauData(data);
    } catch (error) {
      console.error("Failed to fetch MAU data:", error);
    } finally {
      setMauLoading(false);
    }
  };

  const fetchSummaryData = async () => {
    if (!accessToken || !dateValue.from || !dateValue.to) return;

    setSummaryLoading(true);
    try {
      const summary = await userAgentSummaryCall(
        accessToken,
        dateValue.from,
        dateValue.to,
        selectedTags.length > 0 ? selectedTags : undefined,
      );
      setSummaryData(summary);
    } catch (error) {
      console.error("Failed to fetch user agent summary data:", error);
    } finally {
      setSummaryLoading(false);
    }
  };

  useEffect(() => {
    fetchAvailableTags();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [accessToken]);

  useEffect(() => {
    if (!accessToken) return;

    const timeoutId = setTimeout(() => {
      fetchDauData();
      fetchWauData();
      fetchMauData();
    }, 50);

    return () => clearTimeout(timeoutId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [accessToken, userAgentFilter, selectedTags]);

  useEffect(() => {
    if (!dateValue.from || !dateValue.to) return;

    const timeoutId = setTimeout(() => {
      fetchSummaryData();
    }, 50);

    return () => clearTimeout(timeoutId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [accessToken, dateValue, selectedTags]);

  const extractUserAgent = (tag: string): string => {
    if (tag.startsWith("User-Agent: ")) {
      return tag.replace("User-Agent: ", "");
    }
    return tag;
  };

  const truncateUserAgent = (userAgent: string): string => {
    if (userAgent.length > 15) {
      return userAgent.substring(0, 15) + "...";
    }
    return userAgent;
  };

  const getAllTagsForData = (data: TagActiveUsersResponse[]) => {
    const tagTotals = data.reduce(
      (acc, item) => {
        acc[item.tag] = (acc[item.tag] || 0) + item.active_users;
        return acc;
      },
      {} as Record<string, number>,
    );

    return Object.entries(tagTotals)
      .sort(([, a], [, b]) => b - a)
      .map(([tag]) => tag);
  };

  const allDauTags = getAllTagsForData(dauData.results).slice(0, MAX_CATEGORIES);
  const allWauTags = getAllTagsForData(wauData.results).slice(0, MAX_CATEGORIES);
  const allMauTags = getAllTagsForData(mauData.results).slice(0, MAX_CATEGORIES);

  const generateDailyChartData = () => {
    const chartData: any[] = [];
    const endDate = new Date();

    for (let i = 6; i >= 0; i--) {
      const date = new Date(endDate);
      date.setDate(date.getDate() - i);
      const dateStr = date.toISOString().split("T")[0];

      const dayEntry: any = { date: dateStr };

      allDauTags.forEach((tag) => {
        const userAgent = extractUserAgent(tag);
        dayEntry[userAgent] = 0;
      });

      chartData.push(dayEntry);
    }

    dauData.results.forEach((item) => {
      const userAgent = extractUserAgent(item.tag);
      const dayEntry = chartData.find((d) => d.date === item.date);
      if (dayEntry) {
        dayEntry[userAgent] = item.active_users;
      }
    });

    return chartData;
  };

  const dailyChartData = generateDailyChartData();

  const generateWeeklyChartData = () => {
    const chartData: any[] = [];

    for (let weekNum = 1; weekNum <= 7; weekNum++) {
      const weekEntry: any = { week: `Week ${weekNum}` };

      allWauTags.forEach((tag) => {
        const userAgent = extractUserAgent(tag);
        weekEntry[userAgent] = 0;
      });

      chartData.push(weekEntry);
    }

    wauData.results.forEach((item) => {
      const userAgent = extractUserAgent(item.tag);
      const weekMatch = item.date.match(/Week (\d+)/);
      if (weekMatch) {
        const weekLabel = `Week ${weekMatch[1]}`;
        const weekEntry = chartData.find((d) => d.week === weekLabel);
        if (weekEntry) {
          weekEntry[userAgent] = item.active_users;
        }
      }
    });

    return chartData;
  };

  const weeklyChartData = generateWeeklyChartData();

  const generateMonthlyChartData = () => {
    const chartData: any[] = [];

    for (let monthNum = 1; monthNum <= 7; monthNum++) {
      const monthEntry: any = { month: `Month ${monthNum}` };

      allMauTags.forEach((tag) => {
        const userAgent = extractUserAgent(tag);
        monthEntry[userAgent] = 0;
      });

      chartData.push(monthEntry);
    }

    mauData.results.forEach((item) => {
      const userAgent = extractUserAgent(item.tag);
      const monthMatch = item.date.match(/Month (\d+)/);
      if (monthMatch) {
        const monthLabel = `Month ${monthMatch[1]}`;
        const monthEntry = chartData.find((d) => d.month === monthLabel);
        if (monthEntry) {
          monthEntry[userAgent] = item.active_users;
        }
      }
    });

    return chartData;
  };

  const monthlyChartData = generateMonthlyChartData();

  const formatAbbreviatedNumber = (
    value: number,
    decimalPlaces: number = 0,
  ): string => {
    if (value >= 100000000) {
      return (value / 1000000).toFixed(decimalPlaces) + "M";
    } else if (value >= 10000000) {
      return (value / 1000000).toFixed(decimalPlaces) + "M";
    } else if (value >= 1000000) {
      return (value / 1000000).toFixed(decimalPlaces) + "M";
    } else if (value >= 10000) {
      return (value / 1000).toFixed(decimalPlaces) + "K";
    } else if (value >= 1000) {
      return (value / 1000).toFixed(decimalPlaces) + "K";
    } else {
      return value.toFixed(decimalPlaces);
    }
  };

  const tagOptions = availableTags.map((tag) => {
    const userAgent = extractUserAgent(tag);
    const displayName =
      userAgent.length > 50 ? `${userAgent.substring(0, 50)}...` : userAgent;
    return { label: displayName, value: tag, title: userAgent };
  });

  return (
    <div className="space-y-6 mt-6">
      {/* Summary Section Card */}
      <Card className="p-6">
        <div className="space-y-6">
          <div className="flex justify-between items-start">
            <div>
              <h2 className="text-2xl font-semibold m-0">
                Summary by User Agent
              </h2>
              <p className="text-sm text-muted-foreground m-0">
                Performance metrics for different user agents
              </p>
            </div>

            <div className="w-96">
              <Label className="text-sm font-medium block mb-2">
                Filter by User Agents
              </Label>
              <TagMultiSelect
                value={selectedTags}
                onChange={setSelectedTags}
                options={tagOptions}
                placeholder="All User Agents"
              />
            </div>
          </div>

          {/* Top 4 User Agents Cards */}
          {summaryLoading ? (
            <ChartLoader isDateChanging={false} />
          ) : (
            <div className="grid grid-cols-4 gap-4">
              {(summaryData.results || []).slice(0, 4).map((tag, index) => {
                const userAgent = extractUserAgent(tag.tag);
                const displayName = truncateUserAgent(userAgent);
                return (
                  <Card key={index} className="p-4">
                    <TooltipProvider>
                      <ShTooltip>
                        <TooltipTrigger asChild>
                          <h3 className="text-lg font-semibold truncate">
                            {displayName}
                          </h3>
                        </TooltipTrigger>
                        <TooltipContent>{userAgent}</TooltipContent>
                      </ShTooltip>
                    </TooltipProvider>
                    <div className="mt-4 space-y-3">
                      <div>
                        <span className="text-sm text-muted-foreground">
                          Success Requests
                        </span>
                        <div className="text-lg font-semibold">
                          {formatAbbreviatedNumber(tag.successful_requests)}
                        </div>
                      </div>
                      <div>
                        <span className="text-sm text-muted-foreground">
                          Total Tokens
                        </span>
                        <div className="text-lg font-semibold">
                          {formatAbbreviatedNumber(tag.total_tokens)}
                        </div>
                      </div>
                      <div>
                        <span className="text-sm text-muted-foreground">
                          Total Cost
                        </span>
                        <div className="text-lg font-semibold">
                          ${formatAbbreviatedNumber(tag.total_spend, 4)}
                        </div>
                      </div>
                    </div>
                  </Card>
                );
              })}
              {Array.from({
                length: Math.max(0, 4 - (summaryData.results || []).length),
              }).map((_, index) => (
                <Card key={`empty-${index}`} className="p-4">
                  <h3 className="text-lg font-semibold">No Data</h3>
                  <div className="mt-4 space-y-3">
                    <div>
                      <span className="text-sm text-muted-foreground">
                        Success Requests
                      </span>
                      <div className="text-lg font-semibold">-</div>
                    </div>
                    <div>
                      <span className="text-sm text-muted-foreground">
                        Total Tokens
                      </span>
                      <div className="text-lg font-semibold">-</div>
                    </div>
                    <div>
                      <span className="text-sm text-muted-foreground">
                        Total Cost
                      </span>
                      <div className="text-lg font-semibold">-</div>
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          )}
        </div>
      </Card>

      {/* Main Tabs for DAU/WAU/MAU vs Per User Usage */}
      <Card className="p-6">
        <Tabs defaultValue="dau-wau-mau">
          <TabsList className="mb-6">
            <TabsTrigger value="dau-wau-mau">DAU/WAU/MAU</TabsTrigger>
            <TabsTrigger value="per-user">
              Per User Usage (Last 30 Days)
            </TabsTrigger>
          </TabsList>

          <TabsContent value="dau-wau-mau">
            <div className="mb-6">
              <h2 className="text-2xl font-semibold m-0">
                DAU, WAU & MAU per Agent
              </h2>
              <p className="text-sm text-muted-foreground m-0">
                Active users across different time periods
              </p>
            </div>

            <Tabs defaultValue="dau">
              <TabsList className="mb-6">
                <TabsTrigger value="dau">DAU</TabsTrigger>
                <TabsTrigger value="wau">WAU</TabsTrigger>
                <TabsTrigger value="mau">MAU</TabsTrigger>
              </TabsList>

              <TabsContent value="dau">
                <div className="mb-4">
                  <h3 className="text-lg font-semibold">
                    Daily Active Users - Last 7 Days
                  </h3>
                </div>
                {dauLoading ? (
                  <ChartLoader isDateChanging={false} />
                ) : (
                  <BarChart
                    data={dailyChartData}
                    index="date"
                    categories={allDauTags.map(extractUserAgent)}
                    valueFormatter={(value: number) =>
                      formatAbbreviatedNumber(value)
                    }
                    yAxisWidth={60}
                    showLegend={true}
                    stack={true}
                  />
                )}
              </TabsContent>

              <TabsContent value="wau">
                <div className="mb-4">
                  <h3 className="text-lg font-semibold">
                    Weekly Active Users - Last 7 Weeks
                  </h3>
                </div>
                {wauLoading ? (
                  <ChartLoader isDateChanging={false} />
                ) : (
                  <BarChart
                    data={weeklyChartData}
                    index="week"
                    categories={allWauTags.map(extractUserAgent)}
                    valueFormatter={(value: number) =>
                      formatAbbreviatedNumber(value)
                    }
                    yAxisWidth={60}
                    showLegend={true}
                    stack={true}
                  />
                )}
              </TabsContent>

              <TabsContent value="mau">
                <div className="mb-4">
                  <h3 className="text-lg font-semibold">
                    Monthly Active Users - Last 7 Months
                  </h3>
                </div>
                {mauLoading ? (
                  <ChartLoader isDateChanging={false} />
                ) : (
                  <BarChart
                    data={monthlyChartData}
                    index="month"
                    categories={allMauTags.map(extractUserAgent)}
                    valueFormatter={(value: number) =>
                      formatAbbreviatedNumber(value)
                    }
                    yAxisWidth={60}
                    showLegend={true}
                    stack={true}
                  />
                )}
              </TabsContent>
            </Tabs>
          </TabsContent>

          <TabsContent value="per-user">
            <PerUserUsage
              accessToken={accessToken}
              selectedTags={selectedTags}
              formatAbbreviatedNumber={formatAbbreviatedNumber}
            />
          </TabsContent>
        </Tabs>
      </Card>
    </div>
  );
};

export default UserAgentActivity;
