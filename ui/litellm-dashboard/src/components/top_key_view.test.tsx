import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import TopKeyView from "./top_key_view";

describe("TopKeyView", () => {
  it("should render", () => {
    const { container } = render(
      <TopKeyView
        topKeys={[]}
        accessToken={null}
        userID={null}
        userRole={null}
        teams={null}
        premiumUser={false}
        showTags={false}
      />,
    );
    expect(container).toBeTruthy();
  });

  it("should have a table view button", () => {
    const { getByText } = render(
      <TopKeyView
        topKeys={[]}
        accessToken={null}
        userID={null}
        userRole={null}
        teams={null}
        premiumUser={false}
        showTags={false}
      />,
    );
    expect(getByText("Table View")).toBeInTheDocument();
  });

  it("should have a chart view", () => {
    const { getByText } = render(
      <TopKeyView
        topKeys={[]}
        accessToken={null}
        userID={null}
        userRole={null}
        teams={null}
        premiumUser={false}
        showTags={false}
      />,
    );
    expect(getByText("Chart View")).toBeInTheDocument();
  });

  ["Key ID", "Key Alias", "Spend (USD)"].forEach((header) => {
    it(`should have a ${header} column`, () => {
      const { getByText } = render(
        <TopKeyView
          topKeys={[]}
          accessToken={null}
          userID={null}
          userRole={null}
          teams={null}
          premiumUser={false}
          showTags={false}
        />,
      );
      expect(getByText(header)).toBeInTheDocument();
    });
  });

  it("should have a Tags column when showTags is true", () => {
    const { getByText } = render(
      <TopKeyView
        topKeys={[]}
        accessToken={null}
        userID={null}
        userRole={null}
        teams={null}
        premiumUser={false}
        showTags={true}
      />,
    );
    expect(getByText("Tags")).toBeInTheDocument();
  });

  it("should show the key's information on the table", () => {
    const { getByText } = render(
      <TopKeyView
        topKeys={[
          {
            api_key: "key-123",
            key_alias: "Test Key",
            spend: 100,
            tags: [
              { tag: "tag-1", usage: 50 },
              { tag: "tag-2", usage: 30 },
            ],
          },
        ]}
        accessToken="test-token"
        userID={null}
        userRole={null}
        teams={null}
        premiumUser={false}
        showTags={true}
      />,
    );
    expect(getByText("Test Key")).toBeInTheDocument();
    expect(getByText(/tag-1/)).toBeInTheDocument();
    expect(getByText(/tag-2/)).toBeInTheDocument();
    expect(getByText("$100.00")).toBeInTheDocument();
  });
});
