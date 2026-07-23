"use client";

import React, { useMemo, useState } from "react";
import { Info } from "lucide-react";

import AdvancedDatePicker from "@/components/shared/advanced_date_picker";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { CacheLeakageDimension, computeCacheLeakage, pct, usd } from "./costOptimizationUtils";
import { DailyActivityRange } from "./useDailyActivityRange";

interface CacheLeakageCardProps {
  activity: DailyActivityRange;
}

const HeadWithInfo = ({ label, info }: { label: string; info: string }) => (
  <span className="inline-flex items-center gap-1">
    {label}
    <Tooltip>
      <TooltipTrigger render={<span className="inline-flex" aria-label={info} />}>
        <Info className="h-3 w-3 text-gray-400" />
      </TooltipTrigger>
      <TooltipContent className="max-w-xs">{info}</TooltipContent>
    </Tooltip>
  </span>
);

const CacheLeakageCard: React.FC<CacheLeakageCardProps> = ({ activity }) => {
  const { dateValue, onDateChange, results, loading, isFetchingMore } = activity;
  const [dimension, setDimension] = useState<CacheLeakageDimension>("key");
  const leakage = useMemo(() => computeCacheLeakage(results, dimension), [results, dimension]);

  const subject = dimension === "model" ? "Models" : "Keys";
  const firstColumn = dimension === "model" ? "Model" : "Key";
  const emptyNoun = dimension === "model" ? "model" : "key";

  return (
    <TooltipProvider delay={300}>
      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <CardTitle>Cache leakage by {dimension === "model" ? "model" : "virtual key"}</CardTitle>
              <p className="mt-1 text-sm text-muted-foreground">
                {subject} sending large volumes of uncached input with a low cache hit rate are likely missing prompt
                caching. Potential savings is approximate: uncached input priced at the realized cache-read discount.
                {dimension === "model" ? " Limited to Anthropic (Claude) models, which support prompt caching." : ""}
              </p>
            </div>
            <AdvancedDatePicker value={dateValue} onValueChange={onDateChange} />
          </div>
          <Tabs
            value={dimension}
            onValueChange={(value) => setDimension(value === "model" ? "model" : "key")}
            className="mt-3"
          >
            <TabsList>
              <TabsTrigger value="key">By virtual key</TabsTrigger>
              <TabsTrigger value="model">By model</TabsTrigger>
            </TabsList>
          </Tabs>
        </CardHeader>
        <CardContent>
          {leakage.rows.length === 0 ? (
            <p className="py-8 text-center text-sm text-muted-foreground">
              {loading || isFetchingMore ? "Loading..." : `No ${emptyNoun} usage in this range.`}
            </p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>{firstColumn}</TableHead>
                  <TableHead className="text-right">
                    <HeadWithInfo
                      label="Uncached input"
                      info="Input tokens you sent in this range that weren't served from or written to the cache"
                    />
                  </TableHead>
                  <TableHead className="text-right">
                    <HeadWithInfo
                      label="Cache hit rate"
                      info="Share of your input tokens that were served from the cache"
                    />
                  </TableHead>
                  <TableHead className="text-right">
                    <HeadWithInfo
                      label="Potential savings"
                      info="About how much you'd save if this uncached input used prompt caching. Estimated as uncached input tokens times the per-token discount your cached traffic already gets (realized cache savings ÷ cache-read tokens)."
                    />
                  </TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {leakage.rows.map((row) => (
                  <TableRow key={row.id}>
                    <TableCell className="font-medium">
                      {row.label}
                      {row.sublabel && <span className="ml-1 text-xs text-muted-foreground">({row.sublabel})</span>}
                    </TableCell>
                    <TableCell className="text-right">{formatNumberWithCommas(row.uncachedPromptTokens)}</TableCell>
                    <TableCell className="text-right">{pct(row.cacheHitRatio)}</TableCell>
                    <TableCell className="text-right">
                      {row.potentialSavings == null ? "—" : usd(row.potentialSavings)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </TooltipProvider>
  );
};

export default CacheLeakageCard;
