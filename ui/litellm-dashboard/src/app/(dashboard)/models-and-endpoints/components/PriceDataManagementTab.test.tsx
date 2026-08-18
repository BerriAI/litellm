/* @vitest-environment jsdom */
import { render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import PriceDataManagementTab from "./PriceDataManagementTab";

// Deliberately do NOT mock @tremor/react. These tab components render standalone
// (inside antd Tabs / directly as a route page), no longer inside a Tremor
// <TabGroup>. A Tremor <TabPanel> root renders nothing without that context, so
// this asserts the component's content is visible on its own — reverting the root
// back to <TabPanel> makes the title disappear and fails this test.
vi.mock("@/components/price_data_reload", () => ({ default: () => <div>reload</div> }));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: () => ({ accessToken: "sk-test" }) }));
vi.mock("@/app/(dashboard)/hooks/models/useModelCostMap", () => ({
  useModelCostMap: () => ({ refetch: vi.fn() }),
}));

describe("PriceDataManagementTab", () => {
  it("renders its content standalone, without a Tremor TabGroup ancestor", () => {
    const { getByText } = render(<PriceDataManagementTab />);
    expect(getByText("Price Data Management")).toBeInTheDocument();
  });
});
