import { formatNumberWithCommas } from "@/utils/dataUtils";
import { InfoCircleOutlined } from "@ant-design/icons";
import {
  Card,
  Col,
  DonutChart,
  Grid,
  Switch,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
  Title,
} from "@tremor/react";
import { Tooltip } from "antd";
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

const SpendByProvider: React.FC<SpendByProviderProps> = ({ loading, isDateChanging, providerSpend }) => {
  const [includeZeroSpend, setIncludeZeroSpend] = useState(false);
  const [includeUnknown, setIncludeUnknown] = useState(false);

  const filteredProviderSpend = providerSpend.filter((provider) => {
    const isUnknown = provider.provider?.toLowerCase() === "unknown";

    // If includeUnknown is true, always include unknown provider
    if (isUnknown) {
      return includeUnknown;
    }

    // If includeZeroSpend is true, include all providers (including those with 0 spend)
    // Otherwise, only include providers with spend > 0
    if (includeZeroSpend) {
      return true;
    }

    return provider.spend > 0;
  });

  return (
    <Card className="h-full">
      <div className="flex justify-between items-center mb-4">
        <Title>Spend by Provider</Title>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-700">Show Zero Spend</label>
            <Switch checked={includeZeroSpend} onChange={setIncludeZeroSpend} />
          </div>
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1">
              <label className="text-sm text-gray-700">Show Unknown</label>
              <Tooltip title="Requests that failed to route to a provider">
                <InfoCircleOutlined className="text-gray-400 hover:text-gray-600" />
              </Tooltip>
            </div>
            <Switch checked={includeUnknown} onChange={setIncludeUnknown} />
          </div>
        </div>
      </div>
      {loading ? (
        <ChartLoader isDateChanging={isDateChanging} />
      ) : (
        <Grid numItems={2}>
          <Col numColSpan={1}>
            <DonutChart
              className="mt-4 h-40"
              data={filteredProviderSpend}
              index="provider"
              category="spend"
              valueFormatter={(value) => `$${formatNumberWithCommas(value, 2)}`}
              colors={["cyan"]}
            />
          </Col>
          <Col numColSpan={1}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableHeaderCell>Provider</TableHeaderCell>
                  <TableHeaderCell>Spend</TableHeaderCell>
                  <TableHeaderCell className="text-green-600">Successful</TableHeaderCell>
                  <TableHeaderCell className="text-red-600">Failed</TableHeaderCell>
                  <TableHeaderCell>Tokens</TableHeaderCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {filteredProviderSpend.map((provider) => (
                  <TableRow key={provider.provider}>
                    <TableCell>
                      <div className="flex items-center space-x-2">
                        {provider.provider && <ProviderLogo provider={provider.provider} className="w-4 h-4" />}
                        <span>{provider.provider}</span>
                      </div>
                    </TableCell>
                    <TableCell>${formatNumberWithCommas(provider.spend, 2)}</TableCell>
                    <TableCell className="text-green-600">
                      {provider.successful_requests.toLocaleString()}
                    </TableCell>
                    <TableCell className="text-red-600">
                      {provider.failed_requests.toLocaleString()}
                    </TableCell>
                    <TableCell>{provider.tokens.toLocaleString()}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Col>
        </Grid>
      )}
    </Card>
  );
};

export default SpendByProvider;
