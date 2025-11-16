import React, { useState, useEffect } from "react";
import { Card, Col, Grid, Text, Title, Badge } from "@tremor/react";
import { Table, TableHead, TableRow, TableHeaderCell, TableBody, TableCell } from "@tremor/react";
import { customerSpendDetailCall } from "@/components/networking";

interface CustomerDetailViewProps {
  customerId: string;
  accessToken: string | null;
  onBack: () => void;
}

interface ModelSpendData {
  model: string;
  total_spend: number;
  total_requests: number;
  total_tokens: number;
  total_prompt_tokens: number;
  total_completion_tokens: number;
}

interface CustomerDetail {
  end_user_id: string;
  alias: string | null;
  total_spend: number;
  total_requests: number;
  total_tokens: number;
  total_prompt_tokens: number;
  total_completion_tokens: number;
  spend_by_model: ModelSpendData[];
  date_range: {
    start_date: string | null;
    end_date: string | null;
  };
}

const CustomerDetailView: React.FC<CustomerDetailViewProps> = ({ customerId, accessToken, onBack }) => {
  const [customerDetail, setCustomerDetail] = useState<CustomerDetail | null>(null);
  const [loading, setLoading] = useState(false);

  const fetchCustomerDetail = async () => {
    if (!accessToken) return;

    setLoading(true);
    try {
      const response = await customerSpendDetailCall(
        accessToken,
        customerId,
        undefined, // start_date
        undefined // end_date
      );
      setCustomerDetail(response);
    } catch (error) {
      console.error("Error fetching customer detail:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCustomerDetail();
  }, [accessToken, customerId]);

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 2,
      maximumFractionDigits: 6,
    }).format(value);
  };

  const formatNumber = (value: number) => {
    return new Intl.NumberFormat("en-US").format(value);
  };

  if (loading) {
    return (
      <div className="w-full mx-4 h-[75vh] flex items-center justify-center">
        <Text>Loading customer details...</Text>
      </div>
    );
  }

  if (!customerDetail) {
    return (
      <div className="w-full mx-4 h-[75vh] flex items-center justify-center">
        <Text>No customer details found</Text>
      </div>
    );
  }

  return (
    <div className="w-full mx-4 h-[75vh] overflow-y-auto">
      <Grid numItems={1} className="gap-4 p-8 w-full mt-2">
        <Col numColSpan={1}>
          <div className="mb-4">
            <button
              onClick={onBack}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50"
            >
              ← Back to Customer List
            </button>
          </div>

          <Card className="mb-4">
            <div className="flex justify-between items-start mb-4">
              <div>
                <Title>Customer Details</Title>
                <Text className="mt-2">
                  <span className="font-semibold">End User ID:</span> {customerDetail.end_user_id}
                </Text>
                {customerDetail.alias && (
                  <div className="mt-2">
                    <Badge color="blue" size="lg">
                      {customerDetail.alias}
                    </Badge>
                  </div>
                )}
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mt-6">
              <div className="bg-gray-50 p-4 rounded-lg">
                <Text className="text-sm text-gray-600">Total Spend</Text>
                <Title className="mt-1 text-2xl">{formatCurrency(customerDetail.total_spend)}</Title>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <Text className="text-sm text-gray-600">Total Requests</Text>
                <Title className="mt-1 text-2xl">{formatNumber(customerDetail.total_requests)}</Title>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <Text className="text-sm text-gray-600">Total Tokens</Text>
                <Title className="mt-1 text-2xl">{formatNumber(customerDetail.total_tokens)}</Title>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <Text className="text-sm text-gray-600">Prompt Tokens</Text>
                <Title className="mt-1 text-2xl">{formatNumber(customerDetail.total_prompt_tokens)}</Title>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <Text className="text-sm text-gray-600">Completion Tokens</Text>
                <Title className="mt-1 text-2xl">{formatNumber(customerDetail.total_completion_tokens)}</Title>
              </div>
            </div>
          </Card>

          <Card>
            <Title className="mb-4">Usage by Model</Title>
            {customerDetail.spend_by_model.length > 0 ? (
              <Table>
                <TableHead>
                  <TableRow>
                    <TableHeaderCell>Model</TableHeaderCell>
                    <TableHeaderCell className="text-right">Spend</TableHeaderCell>
                    <TableHeaderCell className="text-right">Requests</TableHeaderCell>
                    <TableHeaderCell className="text-right">Total Tokens</TableHeaderCell>
                    <TableHeaderCell className="text-right">Prompt Tokens</TableHeaderCell>
                    <TableHeaderCell className="text-right">Completion Tokens</TableHeaderCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {customerDetail.spend_by_model.map((modelData, index) => (
                    <TableRow key={index}>
                      <TableCell>
                        <Text className="font-medium">{modelData.model}</Text>
                      </TableCell>
                      <TableCell className="text-right">
                        <Text className="font-semibold">{formatCurrency(modelData.total_spend)}</Text>
                      </TableCell>
                      <TableCell className="text-right">
                        <Text>{formatNumber(modelData.total_requests)}</Text>
                      </TableCell>
                      <TableCell className="text-right">
                        <Text>{formatNumber(modelData.total_tokens)}</Text>
                      </TableCell>
                      <TableCell className="text-right">
                        <Text>{formatNumber(modelData.total_prompt_tokens)}</Text>
                      </TableCell>
                      <TableCell className="text-right">
                        <Text>{formatNumber(modelData.total_completion_tokens)}</Text>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            ) : (
              <Text className="text-gray-500 text-center py-8">No model usage data available</Text>
            )}
          </Card>
        </Col>
      </Grid>
    </div>
  );
};

export default CustomerDetailView;
