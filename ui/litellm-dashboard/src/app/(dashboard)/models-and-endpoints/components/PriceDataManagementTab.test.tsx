/* @vitest-environment jsdom */
import { render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import PriceDataManagementTab from "./PriceDataManagementTab";

vi.mock("@/components/price_data_reload", () => ({ default: () => <div>reload</div> }));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: () => ({ accessToken: "sk-test" }) }));
vi.mock("@/app/(dashboard)/hooks/models/useModelCostMap", () => ({
  useModelCostMap: () => ({ refetch: vi.fn() }),
}));

describe("PriceDataManagementTab", () => {
  it("renders its content standalone, without a tab-panel ancestor", () => {
    const { getByText } = render(<PriceDataManagementTab />);
    expect(getByText("Price Data Management")).toBeInTheDocument();
  });
});
