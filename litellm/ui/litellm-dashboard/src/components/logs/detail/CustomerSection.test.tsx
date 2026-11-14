import type { PropsWithChildren } from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { CustomerSection } from "./CustomerSection";

vi.mock("@/components/ui/card", () => ({
  Card: ({ children }: PropsWithChildren) => <section>{children}</section>,
  CardHeader: ({ children }: PropsWithChildren) => <header>{children}</header>,
  CardTitle: ({ children }: PropsWithChildren) => <div>{children}</div>,
  CardContent: ({ children }: PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("./TruncatedValue", () => ({
  TruncatedValue: ({ value }: { value: string }) => <button aria-label={`Copy ${value}`}>{value}</button>,
}));

describe("CustomerSection", () => {
  it("summarizes names when collapsed and links each recognized customer scope when expanded", () => {
    render(
      <CustomerSection
        metadata={{
          user_api_key_customer_org_id: "customer-org-1",
          user_api_key_customer_org_alias: "Customer org display",
          user_api_key_customer_team_id: "customer-team-1",
          user_api_key_customer_team_alias: "Customer team display",
          user_api_key_customer_member_id: "customer-member-1",
          user_api_key_customer_member_alias: "Customer member display",
        }}
      />,
    );

    const trigger = screen.getByRole("button", { name: /Customer org.*Customer org display/ });
    expect(trigger).toHaveAttribute("aria-expanded", "false");
    expect(trigger).toHaveTextContent("Customer member display");
    expect(trigger).toHaveTextContent("Recognized");
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
    fireEvent.click(trigger);
    for (const scope of ["org", "team", "member"]) {
      expect(screen.getByRole("button", { name: `Copy customer-${scope}-1` })).toBeInTheDocument();
      const url = new URL(
        screen.getByRole("link", { name: `View Customer ${scope}` }).getAttribute("href")!,
        "http://localhost",
      );
      expect(url.pathname).toMatch(/customers\/manage$/);
      expect(url.searchParams.get("org")).toBe("customer-org-1");
      expect(url.searchParams.get("team")).toBe(scope === "org" ? null : "customer-team-1");
      expect(url.searchParams.get("member")).toBe(scope === "member" ? "customer-member-1" : null);
    }
  });

  it("shows a customer org without inventing customer team or customer member rows", () => {
    render(<CustomerSection metadata={{ user_api_key_customer_org_id: "customer-org-only" }} />);

    expect(screen.getByText("Customer org")).toBeInTheDocument();
    expect(screen.queryByText("Customer team")).not.toBeInTheDocument();
    expect(screen.queryByText("Customer member")).not.toBeInTheDocument();
  });

  it("displays historical customer IDs without aliases", () => {
    render(<CustomerSection metadata={{ user_api_key_customer_member_id: "historical-customer-member" }} />);

    expect(screen.getByText("Asserted")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Customer member.*historical-customer-member/ }));
    expect(screen.getByRole("button", { name: "Copy historical-customer-member" })).toBeInTheDocument();
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
  });

  it("shows a legacy end user as an asserted Customer without hierarchy links", () => {
    render(<CustomerSection endUser="legacy-customer" />);
    expect(screen.getByText("Asserted")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /legacy-customer/ })).toHaveTextContent("legacy-customer");
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
  });

  it.each([
    undefined,
    null,
    {},
    { user_api_key_team_id: "internal-team" },
    {
      user_api_key_customer_org_id: 42,
      user_api_key_customer_team_id: { invalid: true },
      user_api_key_customer_member_id: " ",
      user_api_key_customer_org_alias: [],
    },
  ])("hides the section without valid customer identity (%j)", (metadata) => {
    render(<CustomerSection metadata={metadata} />);

    expect(screen.queryByText("Customer")).not.toBeInTheDocument();
  });
});
