import React, { useState, useEffect } from "react";
import { Card, Col, Grid, TabPanel, Text, DateRangePickerValue } from "@tremor/react";
import { Icon, Tab, TabGroup, TabList, TabPanels } from "@tremor/react";
import { RefreshIcon } from "@heroicons/react/outline";
import CustomersTable from "@/components/customers/customers_table";
import CustomerDetailView from "@/components/customers/customer_detail_view";
import { customerSpendCall } from "@/components/networking";
import AdvancedDatePicker from "@/components/shared/advanced_date_picker";

interface CustomersViewProps {
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
}

interface CustomerSpendData {
  end_user_id: string;
  alias: string | null;
  total_spend: number;
  total_requests: number;
  total_tokens: number;
  total_prompt_tokens: number;
  total_completion_tokens: number;
}

const CustomersView: React.FC<CustomersViewProps> = ({ accessToken, userRole, userID }) => {
  const [customers, setCustomers] = useState<CustomerSpendData[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedCustomerId, setSelectedCustomerId] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(50);
  const [totalPages, setTotalPages] = useState(0);
  const [totalCustomers, setTotalCustomers] = useState(0);
  const [endUserIdFilter, setEndUserIdFilter] = useState("");
  const [aliasFilter, setAliasFilter] = useState("");
  const [dateValue, setDateValue] = useState<DateRangePickerValue>({
    from: undefined,
    to: undefined,
  });
  const [lastRefreshed, setLastRefreshed] = useState<string>("");

  const fetchCustomers = async () => {
    if (!accessToken) return;

    setLoading(true);
    try {
      // Convert Date objects to YYYY-MM-DD strings for API
      const startDate = dateValue.from
        ? dateValue.from.toISOString().split("T")[0]
        : undefined;
      const endDate = dateValue.to
        ? dateValue.to.toISOString().split("T")[0]
        : undefined;

      const response = await customerSpendCall(
        accessToken,
        startDate,
        endDate,
        endUserIdFilter || undefined,
        aliasFilter || undefined,
        page,
        pageSize
      );

      setCustomers(response.spend_report || []);
      setTotalPages(response.total_pages || 0);
      setTotalCustomers(response.total_customers || 0);
      setLastRefreshed(new Date().toLocaleTimeString());
    } catch (error) {
      console.error("Error fetching customers:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCustomers();
  }, [accessToken, page, pageSize, dateValue]);

  const handleRefresh = () => {
    fetchCustomers();
  };

  const handleFilterApply = () => {
    setPage(1); // Reset to first page when filtering
    fetchCustomers();
  };

  const handleCustomerClick = (customerId: string) => {
    setSelectedCustomerId(customerId);
  };

  const handleBackToList = () => {
    setSelectedCustomerId(null);
  };

  const handleDateChange = (value: DateRangePickerValue) => {
    setDateValue(value);
    setPage(1); // Reset to first page when date range changes
  };

  if (selectedCustomerId) {
    return (
      <CustomerDetailView
        customerId={selectedCustomerId}
        accessToken={accessToken}
        onBack={handleBackToList}
      />
    );
  }

  return (
    <div className="w-full mx-4 h-[75vh]">
      <Grid numItems={1} className="gap-2 p-8 w-full mt-2">
        <Col numColSpan={1} className="flex flex-col gap-2">
          <TabGroup className="gap-2 h-[75vh] w-full">
            <TabList className="flex justify-between mt-2 w-full items-center">
              <div className="flex">
                <Tab>Customer Usage</Tab>
              </div>
              <div className="flex items-center space-x-2">
                {lastRefreshed && <Text>Last Refreshed: {lastRefreshed}</Text>}
                <Icon
                  icon={RefreshIcon}
                  variant="shadow"
                  size="xs"
                  className="self-center cursor-pointer"
                  onClick={handleRefresh}
                />
              </div>
            </TabList>
            <TabPanels>
              <TabPanel>
                <Grid numItems={1} className="gap-2 pt-2 pb-2 h-[75vh] w-full mt-2">
                  <Col numColSpan={1}>
                    <div className="mb-4">
                      <div className="flex items-center gap-3">
                        <AdvancedDatePicker
                          value={dateValue}
                          onValueChange={handleDateChange}
                          label="Filter by Date Range"
                        />
                        {(dateValue.from || dateValue.to) && (
                          <button
                            onClick={() => {
                              setDateValue({ from: undefined, to: undefined });
                              setPage(1);
                            }}
                            className="px-4 py-2 bg-gray-500 text-white text-sm font-medium rounded-md hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-500 transition-colors"
                          >
                            Clear Date Range
                          </button>
                        )}
                      </div>
                    </div>
                    <Card className="w-full mx-auto flex-auto overflow-hidden overflow-y-auto max-h-[65vh]">
                      <div className="border-b px-6 py-3 bg-gray-50">
                        <div className="flex items-center gap-3 justify-end">
                          <div className="w-52">
                            <input
                              type="text"
                              value={endUserIdFilter}
                              onChange={(e) => setEndUserIdFilter(e.target.value)}
                              placeholder="Search by End User ID..."
                              className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                            />
                          </div>
                          <div className="w-52">
                            <input
                              type="text"
                              value={aliasFilter}
                              onChange={(e) => setAliasFilter(e.target.value)}
                              placeholder="Search by Alias..."
                              className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                            />
                          </div>
                          <button
                            onClick={handleFilterApply}
                            className="px-4 py-2 bg-blue-500 text-white text-sm font-medium rounded-md hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors"
                          >
                            Apply
                          </button>
                          {(endUserIdFilter || aliasFilter) && (
                            <button
                              onClick={() => {
                                setEndUserIdFilter("");
                                setAliasFilter("");
                                setPage(1);
                                fetchCustomers();
                              }}
                              className="px-4 py-2 bg-gray-500 text-white text-sm font-medium rounded-md hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-500 transition-colors"
                            >
                              Clear
                            </button>
                          )}
                        </div>
                      </div>
                      <CustomersTable
                        customers={customers}
                        loading={loading}
                        page={page}
                        pageSize={pageSize}
                        totalPages={totalPages}
                        totalCustomers={totalCustomers}
                        onPageChange={setPage}
                        onCustomerClick={handleCustomerClick}
                      />
                    </Card>
                  </Col>
                </Grid>
              </TabPanel>
            </TabPanels>
          </TabGroup>
        </Col>
      </Grid>
    </div>
  );
};

export default CustomersView;
