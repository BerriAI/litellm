import React from "react";
import {
  Card,
  Title,
  Subtitle,
  Table,
  TableHead,
  TableRow,
  TableHeaderCell,
  TableBody,
  TableCell,
  BarChart,
  Text,
  Button,
  Tab,
  TabGroup,
  TabList,
  TabPanel,
  TabPanels,
} from "@tremor/react";

interface UserAgentMetrics {
  dau: number;
  wau: number;
  mau: number;
  successful_requests: number;
  failed_requests: number;
  total_requests: number;
  completed_tokens: number;
  total_tokens: number;
  spend: number;
}

interface UserAgentActivityData {
  date: string;
  tag: string;
  user_agent: string;
  metrics: UserAgentMetrics;
}

interface UserAgentAnalyticsResponse {
  results: UserAgentActivityData[];
  total_count: number;
  page: number;
  page_size: number;
  total_pages: number;
}

interface PerUserUsageProps {
  analyticsData: UserAgentAnalyticsResponse;
  currentPage: number;
  handlePrevPage: () => void;
  handleNextPage: () => void;
  formatAbbreviatedNumber: (value: number, decimalPlaces?: number) => string;
}

const PerUserUsage: React.FC<PerUserUsageProps> = ({
  analyticsData,
  currentPage,
  handlePrevPage,
  handleNextPage,
  formatAbbreviatedNumber,
}) => {
  return (
    <div className="mb-6">
      <Title>Per User Usage</Title>
      <Subtitle>Individual developer usage metrics</Subtitle>
      
      <TabGroup>
        <TabList className="mb-6">
          <Tab>User Details</Tab>
          <Tab>Usage Distribution</Tab>
        </TabList>
        
        <TabPanels>
          {/* Tab 1: Existing User Details Table */}
          <TabPanel>
            <Table>
              <TableHead>
                <TableRow>
                  <TableHeaderCell>User</TableHeaderCell>
                  <TableHeaderCell>User Agent</TableHeaderCell>
                  <TableHeaderCell className="text-right">Success Generations</TableHeaderCell>
                  <TableHeaderCell className="text-right">Total Tokens</TableHeaderCell>
                  <TableHeaderCell className="text-right">Failed Requests</TableHeaderCell>
                  <TableHeaderCell className="text-right">Total Cost</TableHeaderCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {analyticsData.results.slice(0, 10).map((item, index) => (
                  <TableRow key={index}>
                    <TableCell>
                      <div>
                        <div className="font-medium">user_{String(index + 1).padStart(3, '0')}</div>
                        <div className="text-sm text-gray-500">{item.user_agent || "Unknown"}</div>
                      </div>
                    </TableCell>
                    <TableCell>
                      <Text>{item.user_agent || "Unknown"}</Text>
                    </TableCell>
                    <TableCell className="text-right">
                      <Text>{formatAbbreviatedNumber(item.metrics.successful_requests)}</Text>
                    </TableCell>
                    <TableCell className="text-right">
                      <Text>{formatAbbreviatedNumber(item.metrics.total_tokens)}</Text>
                    </TableCell>
                    <TableCell className="text-right">
                      <Text>{formatAbbreviatedNumber(item.metrics.failed_requests)}</Text>
                    </TableCell>
                    <TableCell className="text-right">
                      <Text>${formatAbbreviatedNumber(item.metrics.spend, 4)}</Text>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            
            {analyticsData.results.length > 10 && (
              <div className="mt-4 flex justify-between items-center">
                <Text className="text-sm text-gray-500">
                  Showing 10 of {analyticsData.total_count} results
                </Text>
                <div className="flex gap-2">
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={handlePrevPage}
                    disabled={currentPage === 1}
                  >
                    Previous
                  </Button>
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={handleNextPage}
                    disabled={currentPage >= analyticsData.total_pages}
                  >
                    Next
                  </Button>
                </div>
              </div>
            )}
          </TabPanel>
          
          {/* Tab 2: Usage Distribution Histogram */}
          <TabPanel>
            <div className="mb-4">
              <Title className="text-lg">User Usage Distribution</Title>
              <Subtitle>Number of users by successful request frequency</Subtitle>
            </div>
            
            <BarChart
              data={(() => {
                // Categorize users by successful request count
                const categories = {
                  "1-9 requests": { count: 0, range: [1, 9] },
                  "10-99 requests": { count: 0, range: [10, 99] },
                  "100-999 requests": { count: 0, range: [100, 999] },
                  "1K-9.9K requests": { count: 0, range: [1000, 9999] },
                  "10K-99.9K requests": { count: 0, range: [10000, 99999] },
                  "100K+ requests": { count: 0, range: [100000, Infinity] },
                };
                
                // Count users in each category
                analyticsData.results.forEach(item => {
                  const successCount = item.metrics.successful_requests;
                  
                  Object.entries(categories).forEach(([categoryName, category]) => {
                    if (successCount >= category.range[0] && successCount <= category.range[1]) {
                      category.count++;
                    }
                  });
                });
                
                // Convert to chart data format
                return Object.entries(categories).map(([name, category]) => ({
                  category: name,
                  users: category.count,
                }));
              })()}
              index="category"
              categories={["users"]}
              colors={["blue"]}
              valueFormatter={(value: number) => `${value} users`}
              yAxisWidth={80}
              showLegend={false}
            />
          </TabPanel>
        </TabPanels>
      </TabGroup>
    </div>
  );
};

export default PerUserUsage; 