import { CostEstimateResponse } from "../types";
import { formatNumberWithCommas } from "@/utils/dataUtils";

const formatCostForExport = (value: number | null | undefined): string => {
  if (value === null || value === undefined) return "-";
  if (value === 0) return "$0.00";
  if (value < 0.01) return `$${value.toFixed(6)}`;
  if (value < 1) return `$${value.toFixed(4)}`;
  return `$${formatNumberWithCommas(value, 2)}`;
};

const formatRequestsForExport = (value: number | null | undefined): string => {
  if (value === null || value === undefined) return "-";
  return formatNumberWithCommas(value, 0);
};

export const exportToPDF = (result: CostEstimateResponse): void => {
  const printWindow = window.open("", "_blank");
  if (!printWindow) {
    alert("Please allow popups to export PDF");
    return;
  }

  const html = `
    <!DOCTYPE html>
    <html>
    <head>
      <title>Cost Estimate Report - ${result.model}</title>
      <style>
        body {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          padding: 40px;
          max-width: 800px;
          margin: 0 auto;
          color: #333;
        }
        h1 {
          color: #1a1a1a;
          border-bottom: 2px solid #1890ff;
          padding-bottom: 10px;
          margin-bottom: 30px;
        }
        h2 {
          color: #444;
          margin-top: 30px;
          margin-bottom: 15px;
        }
        .meta {
          background: #f5f5f5;
          padding: 15px;
          border-radius: 8px;
          margin-bottom: 30px;
        }
        .meta p {
          margin: 5px 0;
        }
        table {
          width: 100%;
          border-collapse: collapse;
          margin-bottom: 20px;
        }
        th, td {
          padding: 12px;
          text-align: left;
          border-bottom: 1px solid #ddd;
        }
        th {
          background: #f8f9fa;
          font-weight: 600;
        }
        .cost-value {
          font-family: monospace;
          font-size: 14px;
        }
        .total-row {
          font-weight: bold;
          background: #e6f7ff;
        }
        .footer {
          margin-top: 40px;
          padding-top: 20px;
          border-top: 1px solid #ddd;
          font-size: 12px;
          color: #666;
        }
        @media print {
          body { padding: 20px; }
        }
      </style>
    </head>
    <body>
      <h1>🚅 LiteLLM Cost Estimate Report</h1>
      
      <div class="meta">
        <p><strong>Model:</strong> ${result.model}</p>
        ${result.provider ? `<p><strong>Provider:</strong> ${result.provider}</p>` : ""}
        <p><strong>Input Tokens per Request:</strong> ${formatRequestsForExport(result.input_tokens)}</p>
        <p><strong>Output Tokens per Request:</strong> ${formatRequestsForExport(result.output_tokens)}</p>
        ${result.num_requests_per_day ? `<p><strong>Requests per Day:</strong> ${formatRequestsForExport(result.num_requests_per_day)}</p>` : ""}
        ${result.num_requests_per_month ? `<p><strong>Requests per Month:</strong> ${formatRequestsForExport(result.num_requests_per_month)}</p>` : ""}
      </div>

      <h2>Per-Request Cost Breakdown</h2>
      <table>
        <tr>
          <th>Cost Type</th>
          <th>Amount</th>
        </tr>
        <tr>
          <td>Input Cost</td>
          <td class="cost-value">${formatCostForExport(result.input_cost_per_request)}</td>
        </tr>
        <tr>
          <td>Output Cost</td>
          <td class="cost-value">${formatCostForExport(result.output_cost_per_request)}</td>
        </tr>
        <tr>
          <td>Margin/Fee</td>
          <td class="cost-value">${formatCostForExport(result.margin_cost_per_request)}</td>
        </tr>
        <tr class="total-row">
          <td>Total per Request</td>
          <td class="cost-value">${formatCostForExport(result.cost_per_request)}</td>
        </tr>
      </table>

      ${result.daily_cost !== null ? `
      <h2>Daily Cost Estimate (${formatRequestsForExport(result.num_requests_per_day)} requests/day)</h2>
      <table>
        <tr>
          <th>Cost Type</th>
          <th>Amount</th>
        </tr>
        <tr>
          <td>Input Cost</td>
          <td class="cost-value">${formatCostForExport(result.daily_input_cost)}</td>
        </tr>
        <tr>
          <td>Output Cost</td>
          <td class="cost-value">${formatCostForExport(result.daily_output_cost)}</td>
        </tr>
        <tr>
          <td>Margin/Fee</td>
          <td class="cost-value">${formatCostForExport(result.daily_margin_cost)}</td>
        </tr>
        <tr class="total-row">
          <td>Total Daily</td>
          <td class="cost-value">${formatCostForExport(result.daily_cost)}</td>
        </tr>
      </table>
      ` : ""}

      ${result.monthly_cost !== null ? `
      <h2>Monthly Cost Estimate (${formatRequestsForExport(result.num_requests_per_month)} requests/month)</h2>
      <table>
        <tr>
          <th>Cost Type</th>
          <th>Amount</th>
        </tr>
        <tr>
          <td>Input Cost</td>
          <td class="cost-value">${formatCostForExport(result.monthly_input_cost)}</td>
        </tr>
        <tr>
          <td>Output Cost</td>
          <td class="cost-value">${formatCostForExport(result.monthly_output_cost)}</td>
        </tr>
        <tr>
          <td>Margin/Fee</td>
          <td class="cost-value">${formatCostForExport(result.monthly_margin_cost)}</td>
        </tr>
        <tr class="total-row">
          <td>Total Monthly</td>
          <td class="cost-value">${formatCostForExport(result.monthly_cost)}</td>
        </tr>
      </table>
      ` : ""}

      ${result.input_cost_per_token || result.output_cost_per_token ? `
      <h2>Token Pricing</h2>
      <table>
        <tr>
          <th>Token Type</th>
          <th>Price per 1M Tokens</th>
        </tr>
        ${result.input_cost_per_token ? `
        <tr>
          <td>Input Tokens</td>
          <td class="cost-value">$${(result.input_cost_per_token * 1000000).toFixed(2)}</td>
        </tr>
        ` : ""}
        ${result.output_cost_per_token ? `
        <tr>
          <td>Output Tokens</td>
          <td class="cost-value">$${(result.output_cost_per_token * 1000000).toFixed(2)}</td>
        </tr>
        ` : ""}
      </table>
      ` : ""}

      <div class="footer">
        <p>Generated by LiteLLM Pricing Calculator on ${new Date().toLocaleString()}</p>
      </div>
    </body>
    </html>
  `;

  printWindow.document.write(html);
  printWindow.document.close();
  printWindow.onload = () => {
    printWindow.print();
  };
};

export const exportToCSV = (result: CostEstimateResponse): void => {
  const rows = [
    ["🚅 LiteLLM Cost Estimate Report"],
    [""],
    ["Configuration"],
    ["Model", result.model],
    ["Provider", result.provider || "-"],
    ["Input Tokens per Request", result.input_tokens.toString()],
    ["Output Tokens per Request", result.output_tokens.toString()],
    ["Requests per Day", result.num_requests_per_day?.toString() || "-"],
    ["Requests per Month", result.num_requests_per_month?.toString() || "-"],
    [""],
    ["Per-Request Costs"],
    ["Input Cost", result.input_cost_per_request.toString()],
    ["Output Cost", result.output_cost_per_request.toString()],
    ["Margin/Fee", result.margin_cost_per_request.toString()],
    ["Total per Request", result.cost_per_request.toString()],
  ];

  if (result.daily_cost !== null) {
    rows.push(
      [""],
      ["Daily Costs"],
      ["Daily Input Cost", result.daily_input_cost?.toString() || "-"],
      ["Daily Output Cost", result.daily_output_cost?.toString() || "-"],
      ["Daily Margin/Fee", result.daily_margin_cost?.toString() || "-"],
      ["Total Daily", result.daily_cost.toString()]
    );
  }

  if (result.monthly_cost !== null) {
    rows.push(
      [""],
      ["Monthly Costs"],
      ["Monthly Input Cost", result.monthly_input_cost?.toString() || "-"],
      ["Monthly Output Cost", result.monthly_output_cost?.toString() || "-"],
      ["Monthly Margin/Fee", result.monthly_margin_cost?.toString() || "-"],
      ["Total Monthly", result.monthly_cost.toString()]
    );
  }

  if (result.input_cost_per_token || result.output_cost_per_token) {
    rows.push(
      [""],
      ["Token Pricing (per 1M tokens)"],
      ["Input Token Price", result.input_cost_per_token ? `$${(result.input_cost_per_token * 1000000).toFixed(2)}` : "-"],
      ["Output Token Price", result.output_cost_per_token ? `$${(result.output_cost_per_token * 1000000).toFixed(2)}` : "-"]
    );
  }

  const csv = rows.map(row => row.join(",")).join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `cost_estimate_${result.model.replace(/\//g, "_")}_${new Date().toISOString().split("T")[0]}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  window.URL.revokeObjectURL(url);
};

