import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, testQueryClient } from "../../../tests/test-utils";
import type { EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";
import CustomerMultiSelect from "./CustomerMultiSelect";

const customer = (number: number): EndUser => ({
  user_id: `customer-${number}`,
  alias: `Customer ${number}`,
  spend: 0,
  blocked: false,
});

const requests: URL[] = [];
let failList = false;

function Controlled({ initialValue = [] }: { initialValue?: string[] }) {
  const [value, setValue] = useState(initialValue);
  return (
    <>
      <CustomerMultiSelect value={value} onChange={setValue} />
      <output aria-label="Selected customers">{value.join(",")}</output>
    </>
  );
}

beforeEach(() => {
  testQueryClient.clear();
  requests.length = 0;
  failList = false;
  const encode = (value: object) => btoa(JSON.stringify(value)).replaceAll("=", "");
  document.cookie = `token=${encode({ alg: "none" })}.${encode({ key: "sk-test", exp: Date.now() / 1000 + 3600 })}.test; Path=/`;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = new URL(input instanceof Request ? input.url : String(input), "http://localhost");
      if (url.pathname.endsWith("/litellm-ui-config")) {
        return Response.json({ server_root_path: "", proxy_base_url: "" });
      }
      if (url.pathname === "/management/v1/users/me") {
        return Response.json({ user_id: "admin", user_role: "proxy_admin" });
      }
      requests.push(url);
      if (url.pathname === "/customer/info") {
        const id = url.searchParams.get("end_user_id") ?? "";
        return Response.json(customer(Number(id.replace("customer-", ""))));
      }
      if (url.pathname === "/customer/list") {
        if (failList) return Response.json({ detail: "Unavailable" }, { status: 503 });
        const search = url.searchParams.get("search");
        const page = Number(url.searchParams.get("page"));
        if (search) return Response.json(search.includes("5000") ? [customer(5000)] : []);
        return Response.json(
          page === 1 ? Array.from({ length: 20 }, (_, index) => customer(index + 1)) : [customer(21)],
        );
      }
      return Response.json({});
    }),
  );
});

afterEach(() => {
  testQueryClient.clear();
  vi.unstubAllGlobals();
  document.cookie = "token=; Max-Age=0; Path=/";
});

describe("CustomerMultiSelect", () => {
  it("loads a bounded page, appends on scroll, searches beyond it and preserves multiple selections", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Controlled />);
    const input = screen.getByRole("combobox");
    await user.click(input);
    await screen.findByRole("option", { name: /^Customer 1\s*customer-1$/ });
    const firstRequest = requests.find((url) => url.pathname === "/customer/list");
    expect(firstRequest?.searchParams.get("page_size")).toBe("20");
    expect(firstRequest?.searchParams.get("page")).toBe("1");
    expect(requests.filter((url) => url.pathname === "/customer/list")).toHaveLength(1);

    const list = screen.getByTestId("paginated-multi-select-list");
    Object.defineProperties(list, {
      scrollHeight: { value: 1000, configurable: true },
      clientHeight: { value: 200, configurable: true },
    });
    fireEvent.scroll(list, { target: { scrollTop: 800 } });
    await user.click(await screen.findByRole("option", { name: /^Customer 21\s*customer-21$/ }));
    expect(screen.getByLabelText("Selected customers")).toHaveTextContent("customer-21");

    await user.type(input, "5000");
    await user.click(await screen.findByRole("option", { name: /^Customer 5000\s*customer-5000$/ }));
    expect(screen.getByLabelText("Selected customers")).toHaveTextContent("customer-21,customer-5000");
    expect(screen.getByLabelText("Customer 21")).toBeInTheDocument();
    const searchRequest = requests.find((url) => url.searchParams.get("search") === "5000");
    expect(searchRequest?.searchParams.get("page")).toBe("1");

    await user.click(screen.getByRole("button", { name: "Clear all customers" }));
    expect(screen.getByLabelText("Selected customers")).toBeEmptyDOMElement();
  });

  it("resolves a deep-linked selection by ID without fetching the directory", async () => {
    renderWithProviders(<Controlled initialValue={["customer-5000"]} />);
    expect(await screen.findByLabelText("Customer 5000")).toBeInTheDocument();
    expect(
      requests.filter((url) => url.pathname === "/customer/info").map((url) => url.searchParams.get("end_user_id")),
    ).toEqual(["customer-5000"]);
    expect(requests.filter((url) => url.pathname === "/customer/list")).toHaveLength(1);
    expect(screen.getByLabelText("Selected customers")).toHaveTextContent("customer-5000");
  });

  it("distinguishes a failed request from an empty directory", async () => {
    failList = true;
    renderWithProviders(<Controlled />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Unable to load customers");
    expect(screen.queryByText("No customers found")).not.toBeInTheDocument();
  });

  it("shows an empty result after a successful search", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Controlled />);
    const input = screen.getByRole("combobox");
    await user.click(input);
    await screen.findByRole("option", { name: /^Customer 1\s*customer-1$/ });
    await user.type(input, "missing");
    expect(await screen.findByText(/No customers found/)).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    await waitFor(() => expect(requests.some((url) => url.searchParams.get("search") === "missing")).toBe(true));
  });
});
