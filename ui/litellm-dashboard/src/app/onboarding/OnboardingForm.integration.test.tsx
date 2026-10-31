import { setGlobalLitellmHeaderName } from "@/components/networking";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { OnboardingForm } from "./OnboardingForm";

vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams("invitation_id=inv-123"),
}));

afterEach(() => {
  setGlobalLitellmHeaderName("Authorization");
  sessionStorage.clear();
  vi.unstubAllGlobals();
});

describe.each(["signup", "reset_password"] as const)("OnboardingForm %s authentication", (variant) => {
  it.each([
    { name: "custom header", authHeader: "x-litellm-api-key", previousHeader: "Authorization" },
    { name: "default header", authHeader: "Authorization", previousHeader: "x-stale-auth" },
    { name: "missing header claim", authHeader: undefined, previousHeader: "x-stale-auth" },
  ])("submits with the token's $name instead of prior session state", async ({ authHeader, previousHeader }) => {
    const expectedHeader = authHeader ?? "Authorization";
    const onboardingClaims = {
      user_email: "alice@example.test",
      user_id: "user-123",
      key: "onboarding-key",
      auth_header_name: authHeader,
    };
    const token = `e30.${btoa(JSON.stringify(onboardingClaims))}.signature`;
    const claims: Request[] = [];
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
    });
    setGlobalLitellmHeaderName(previousHeader);
    vi.stubGlobal("location", {
      ...window.location,
      href: "http://localhost/ui/onboarding/?invitation_id=inv-123",
      pathname: "/ui/onboarding/",
    });
    vi.stubGlobal(
      "fetch",
      vi.fn<typeof fetch>(async (input, init) => {
        const url = new URL(input instanceof Request ? input.url : String(input), "http://localhost");
        switch (url.pathname) {
          case "/ui/litellm-runtime-config.json":
            return Response.json({});
          case "/litellm/.well-known/litellm-ui-config":
            return Response.json({ server_root_path: "", proxy_base_url: null });
          case "/onboarding/get_token":
            return Response.json({ token });
          case "/onboarding/claim_token": {
            const claim = new Request(url, init);
            claims.push(claim);
            return claim.headers.get(expectedHeader) === "Bearer onboarding-key"
              ? Response.json({ token: "new-session-token" })
              : Response.json(
                  { detail: { error: "Missing onboarding session for invitation link." } },
                  { status: 401 },
                );
          }
          default:
            throw new Error(`Unexpected request ${url.pathname}`);
        }
      }),
    );

    try {
      render(
        <QueryClientProvider client={queryClient}>
          <OnboardingForm variant={variant} />
        </QueryClientProvider>,
      );
      await waitFor(() => expect(screen.getByLabelText("Email Address")).toHaveValue("alice@example.test"));
      fireEvent.change(screen.getByLabelText("Password"), { target: { value: "New-password-for-test" } });
      fireEvent.click(screen.getByRole("button", { name: variant === "signup" ? "Sign Up" : "Reset Password" }));

      await waitFor(() => expect(claims).toHaveLength(1));
      expect(claims[0].headers.get(expectedHeader)).toBe("Bearer onboarding-key");
      expect(claims[0].headers.has(previousHeader)).toBe(false);
      expect(await claims[0].json()).toEqual({
        invitation_link: "inv-123",
        user_id: "user-123",
        password: "New-password-for-test",
      });
      await waitFor(() => expect(sessionStorage.getItem("token")).toBe("new-session-token"));
    } finally {
      queryClient.clear();
    }
  });
});
