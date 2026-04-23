import { formatNumberWithCommas } from "@/utils/dataUtils";
import { Card } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
// eslint-disable-next-line litellm-ui/no-banned-ui-imports
import { DonutChart } from "@tremor/react";
import { Info } from "lucide-react";
import React, { useState } from "react";
import { ProviderLogo } from "../../../molecules/models/ProviderLogo";
import { ChartLoader } from "../../../shared/chart_loader";

interface ProviderSpendData {
  provider: string;
  spend: number;
  requests: number;
  successful_requests: number;
  failed_requests: number;
  tokens: number;
}

interface SpendByProviderProps {
  loading: boolean;
  isDateChanging: boolean;
  providerSpend: ProviderSpendData[];
}

const SpendByProvider: React.FC<SpendByProviderProps> = ({
  loading,
  isDateChanging,
  providerSpend,
}) => {
  const [includeZeroSpend, setIncludeZeroSpend] = useState(false);
  const [includeUnknown, setIncludeUnknown] = useState(false);

  const filteredProviderSpend = providerSpend.filter((provider) => {
    const isUnknown = provider.provider?.toLowerCase() === "unknown";

    if (isUnknown) {
      return includeUnknown;
    }

    if (includeZeroSpend) {
      return true;
    }

    return provider.spend > 0;
  });

  return (
    <Card className="h-full p-4">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold">Spend by Provider</h3>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <label className="text-sm text-foreground">Show Zero Spend</label>
            <Switch
              checked={includeZeroSpend}
              onCheckedChange={setIncludeZeroSpend}
            />
          </div>
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1">
              <label className="text-sm text-foreground">Show Unknown</label>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Info className="h-3 w-3 text-muted-foreground" />
                  </TooltipTrigger>
                  <TooltipContent>
                    Requests that failed to route to a provider
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
            <Switch
              checked={includeUnknown}
              onCheckedChange={setIncludeUnknown}
            />
          </div>
        </div>
      </div>
      {loading ? (
        <ChartLoader isDateChanging={isDateChanging} />
      ) : (
        <div className="grid grid-cols-2 gap-4">
          <div>
            <DonutChart
              className="mt-4 h-40"
              data={filteredProviderSpend}
              index="provider"
              category="spend"
              valueFormatter={(value) =>
                `$${formatNumberWithCommas(value, 2)}`
              }
              colors={["cyan"]}
            />
          </div>
          <div>
            <div className="border border-border rounded-md overflow-hidden">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Provider</TableHead>
                    <TableHead>Spend</TableHead>
                    <TableHead className="text-emerald-600 dark:text-emerald-400">
                      Successful
                    </TableHead>
                    <TableHead className="text-destructive">Failed</TableHead>
                    <TableHead>Tokens</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredProviderSpend.map((provider) => (
                    <TableRow key={provider.provider}>
                      <TableCell>
                        <div className="flex items-center space-x-2">
                          {provider.provider && (
                            <ProviderLogo
                              provider={provider.provider}
                              className="w-4 h-4"
                            />
                          )}
                          <span>{provider.provider}</span>
                        </div>
                      </TableCell>
                      <TableCell>
                        ${formatNumberWithCommas(provider.spend, 2)}
                      </TableCell>
                      <TableCell className="text-emerald-600 dark:text-emerald-400">
                        {provider.successful_requests.toLocaleString()}
                      </TableCell>
                      <TableCell className="text-destructive">
                        {provider.failed_requests.toLocaleString()}
                      </TableCell>
                      <TableCell>
                        {provider.tokens.toLocaleString()}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </div>
        </div>
      )}
    </Card>
  );
};

export default SpendByProvider;
