import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { describe, expect, it, vi } from "vitest";
import ModelRetrySettingsTab from "./ModelRetrySettingsTab";

// TabPanel requires a parent Tabs context in Tremor. We stub it to render children
// directly so the component can be tested in isolation.
vi.mock("@tremor/react", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@tremor/react")>();
  // Re-apply the global Button/Tooltip overrides from tests/setupTests.ts. A file-level
  // vi.mock fully replaces the setup-level mock, so without this the real Tremor Button
  // leaks through and its useTooltip(300) schedules a native setTimeout that can fire
  // post-teardown -> "window is not defined".
  return {
    ...actual,
    TabPanel: ({ children }: { children: React.ReactNode }) => React.createElement("div", null, children),
    Button: React.forwardRef<HTMLButtonElement, any>(({ children, variant, size, loading, ...props }, ref) =>
      React.createElement("button", { ...props, ref }, children),
    ),
    Tooltip: ({ children }: { children?: React.ReactNode }) => React.createElement(React.Fragment, null, children),
    // Keep Select/SelectItem as the real implementation so scope-switching is testable
  };
});

type GlobalRetryPolicy = { [key: string]: number };
type ModelGroupRetryPolicy = { [key: string]: { [key: string]: number } | undefined };

const DEFAULT_RETRY = 0;

const buildProps = (overrides: Record<string, unknown> = {}) => ({
  selectedModelGroup: "global" as string | null,
  setSelectedModelGroup: vi.fn(),
  availableModelGroups: ["gpt-4", "claude-3-opus"],
  globalRetryPolicy: null as GlobalRetryPolicy | null,
  setGlobalRetryPolicy: vi.fn(),
  defaultRetry: DEFAULT_RETRY,
  modelGroupRetryPolicy: null as ModelGroupRetryPolicy | null,
  setModelGroupRetryPolicy: vi.fn(),
  handleSaveRetrySettings: vi.fn(),
  ...overrides,
});

describe("ModelRetrySettingsTab", () => {
  it("should render the 'Global Retry Policy' heading when selectedModelGroup is 'global'", () => {
    render(<ModelRetrySettingsTab {...buildProps()} />);

    expect(screen.getByText("Global Retry Policy")).toBeInTheDocument();
  });

  it("should render a model-specific heading when a model group is selected", () => {
    render(<ModelRetrySettingsTab {...buildProps({ selectedModelGroup: "gpt-4" })} />);

    expect(screen.getByText("Retry Policy for gpt-4")).toBeInTheDocument();
  });

  it("should render a row for every error type in the retry policy map", () => {
    render(<ModelRetrySettingsTab {...buildProps()} />);

    expect(screen.getByText(/BadRequestError \(400\)/)).toBeInTheDocument();
    expect(screen.getByText(/AuthenticationError/)).toBeInTheDocument();
    expect(screen.getByText(/TimeoutError \(408\)/)).toBeInTheDocument();
    expect(screen.getByText(/RateLimitError \(429\)/)).toBeInTheDocument();
    expect(screen.getByText(/ContentPolicyViolationError \(400\)/)).toBeInTheDocument();
    expect(screen.getByText(/InternalServerError \(500\)/)).toBeInTheDocument();
  });

  it("should use defaultRetry when globalRetryPolicy is null (global scope)", () => {
    render(<ModelRetrySettingsTab {...buildProps({ defaultRetry: 3 })} />);

    // All 6 spinbutton inputs should show the defaultRetry value
    const inputs = screen.getAllByRole("spinbutton");
    inputs.forEach((input) => {
      expect(input).toHaveValue("3");
    });
  });

  it("should show globalRetryPolicy values when they are set (global scope)", () => {
    const globalRetryPolicy: GlobalRetryPolicy = {
      RateLimitErrorRetries: 5,
    };
    render(<ModelRetrySettingsTab {...buildProps({ globalRetryPolicy, defaultRetry: 0 })} />);

    // The RateLimitError row is the 4th entry in the map
    const inputs = screen.getAllByRole("spinbutton");
    const rateLimitInput = inputs[3]; // 0-indexed: Bad(0), Auth(1), Timeout(2), Rate(3)
    expect(rateLimitInput).toHaveValue("5");

    // Unset entries fall back to defaultRetry (0)
    expect(inputs[0]).toHaveValue("0");
  });

  it("should leave model-scope rows empty with the inherited value as placeholder when there is no override", () => {
    const globalRetryPolicy: GlobalRetryPolicy = {
      TimeoutErrorRetries: 7,
    };
    render(
      <ModelRetrySettingsTab
        {...buildProps({
          selectedModelGroup: "gpt-4",
          globalRetryPolicy,
          modelGroupRetryPolicy: null,
          defaultRetry: 1,
        })}
      />,
    );

    // No override exists, so the input is empty and the inherited value is only
    // a placeholder -- this is what keeps 0 ("zero retries") distinct from
    // "inherit the global value".
    const inputs = screen.getAllByRole("spinbutton");
    expect(inputs[2]).toHaveValue(""); // TimeoutError row
    expect(inputs[2]).toHaveAttribute("placeholder", "7"); // inherited from global
    expect(inputs[0]).toHaveValue(""); // BadRequestError row (no global)
    expect(inputs[0]).toHaveAttribute("placeholder", "1"); // inherited from defaultRetry
  });

  it("should clear a model-group override when Reset is clicked", async () => {
    const user = userEvent.setup();
    const setModelGroupRetryPolicy = vi.fn();
    render(
      <ModelRetrySettingsTab
        {...buildProps({
          selectedModelGroup: "gpt-4",
          modelGroupRetryPolicy: { "gpt-4": { BadRequestErrorRetries: 5 } },
          setModelGroupRetryPolicy,
          defaultRetry: 0,
        })}
      />,
    );

    // Reset only renders for rows that actually have an override
    const resetButtons = screen.getAllByRole("button", { name: /reset/i });
    expect(resetButtons).toHaveLength(1);

    await user.click(resetButtons[0]);

    const updater = setModelGroupRetryPolicy.mock.calls.at(-1)![0];
    const result = updater({ "gpt-4": { BadRequestErrorRetries: 5 } });
    expect(result["gpt-4"]).not.toHaveProperty("BadRequestErrorRetries");
  });

  it("should disable the Save button while a save is in flight", () => {
    render(<ModelRetrySettingsTab {...buildProps({ isSaving: true })} />);

    expect(screen.getByRole("button", { name: /save/i })).toBeDisabled();
  });

  it("should prefer model-specific retry count over the global value (model scope)", () => {
    const globalRetryPolicy: GlobalRetryPolicy = {
      RateLimitErrorRetries: 3,
    };
    const modelGroupRetryPolicy: ModelGroupRetryPolicy = {
      "gpt-4": { RateLimitErrorRetries: 9 },
    };
    render(
      <ModelRetrySettingsTab
        {...buildProps({
          selectedModelGroup: "gpt-4",
          globalRetryPolicy,
          modelGroupRetryPolicy,
          defaultRetry: 0,
        })}
      />,
    );

    // The model-specific value (9) should win over global (3)
    const inputs = screen.getAllByRole("spinbutton");
    expect(inputs[3]).toHaveValue("9");
  });

  it("should show the global reference value text for each row in model-specific scope", () => {
    const globalRetryPolicy: GlobalRetryPolicy = { BadRequestErrorRetries: 2 };
    render(
      <ModelRetrySettingsTab
        {...buildProps({
          selectedModelGroup: "gpt-4",
          globalRetryPolicy,
          defaultRetry: 0,
        })}
      />,
    );

    // "(Global: X)" annotations are shown next to each row label in model scope
    expect(screen.getByText("(Global: 2)")).toBeInTheDocument();
  });

  it("should not show global reference annotations in global scope", () => {
    render(<ModelRetrySettingsTab {...buildProps({ selectedModelGroup: "global" })} />);

    expect(screen.queryByText(/Global:/)).not.toBeInTheDocument();
  });

  it("should call handleSaveRetrySettings when the Save button is clicked", async () => {
    const user = userEvent.setup();
    const handleSaveRetrySettings = vi.fn();
    render(<ModelRetrySettingsTab {...buildProps({ handleSaveRetrySettings })} />);

    await user.click(screen.getByRole("button", { name: /save/i }));

    expect(handleSaveRetrySettings).toHaveBeenCalledTimes(1);
  });

  it("should call setGlobalRetryPolicy with an updater function when an input changes (global scope)", async () => {
    const user = userEvent.setup();
    const setGlobalRetryPolicy = vi.fn();
    render(
      <ModelRetrySettingsTab
        {...buildProps({
          selectedModelGroup: "global",
          globalRetryPolicy: { BadRequestErrorRetries: 0 },
          setGlobalRetryPolicy,
          defaultRetry: 0,
        })}
      />,
    );

    const inputs = screen.getAllByRole("spinbutton");
    await user.clear(inputs[0]);
    await user.type(inputs[0], "4");

    // setGlobalRetryPolicy is called with a function updater
    expect(setGlobalRetryPolicy).toHaveBeenCalled();
    const updater = setGlobalRetryPolicy.mock.calls.at(-1)![0];
    expect(typeof updater).toBe("function");

    // Calling the updater returns the merged policy
    const result = updater({ BadRequestErrorRetries: 0 });
    expect(result).toMatchObject({ BadRequestErrorRetries: 4 });
  });

  it("should call setModelGroupRetryPolicy with an updater function when an input changes (model scope)", async () => {
    const user = userEvent.setup();
    const setModelGroupRetryPolicy = vi.fn();
    render(
      <ModelRetrySettingsTab
        {...buildProps({
          selectedModelGroup: "gpt-4",
          modelGroupRetryPolicy: { "gpt-4": { BadRequestErrorRetries: 0 } },
          setModelGroupRetryPolicy,
          defaultRetry: 0,
        })}
      />,
    );

    const inputs = screen.getAllByRole("spinbutton");
    await user.clear(inputs[0]);
    await user.type(inputs[0], "2");

    expect(setModelGroupRetryPolicy).toHaveBeenCalled();
    const updater = setModelGroupRetryPolicy.mock.calls.at(-1)![0];
    expect(typeof updater).toBe("function");

    // Calling the updater returns the merged model-group policy
    const result = updater({ "gpt-4": { BadRequestErrorRetries: 0 } });
    expect(result["gpt-4"]).toMatchObject({ BadRequestErrorRetries: 2 });
  });
});
