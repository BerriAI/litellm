import { useState } from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { getMajorAirlines } from "@/components/networking";

import CompetitorIntentConfiguration, { type CompetitorIntentConfig } from "./CompetitorIntentConfiguration";

vi.mock("@/components/networking", () => ({ getMajorAirlines: vi.fn() }));

const mockAirlines = vi.mocked(getMajorAirlines);
const onChange = vi.fn();

const DEFAULT_CONFIG: CompetitorIntentConfig = {
  competitor_intent_type: "airline",
  brand_self: [],
  locations: [],
  policy: {
    competitor_comparison: "refuse",
    possible_competitor_comparison: "reframe",
  },
  threshold_high: 0.7,
  threshold_medium: 0.45,
  threshold_low: 0.3,
};

const Harness = ({ initialEnabled = true }: { initialEnabled?: boolean }) => {
  const [enabled, setEnabled] = useState(initialEnabled);
  const [config, setConfig] = useState<CompetitorIntentConfig | null>(initialEnabled ? DEFAULT_CONFIG : null);
  const handleChange = (nextEnabled: boolean, nextConfig: CompetitorIntentConfig | null) => {
    onChange(nextEnabled, nextConfig);
    setEnabled(nextEnabled);
    setConfig(nextConfig);
  };
  return (
    <CompetitorIntentConfiguration enabled={enabled} config={config} accessToken="sk-test" onChange={handleChange} />
  );
};

const lastConfig = (): CompetitorIntentConfig => onChange.mock.calls[onChange.mock.calls.length - 1][1];

const chooseOption = async (user: ReturnType<typeof userEvent.setup>, index: number, optionText: string) => {
  await user.click(screen.getAllByRole("combobox")[index]);
  await user.click(await screen.findByRole("option", { name: optionText }));
};

describe("CompetitorIntentConfiguration reported config", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockAirlines.mockResolvedValue({ airlines: [] });
  });

  it("reports the seeded config when switched on and null when switched off", async () => {
    const user = userEvent.setup();
    render(<Harness initialEnabled={false} />);

    await user.click(screen.getByRole("switch"));
    expect(onChange).toHaveBeenNthCalledWith(1, true, DEFAULT_CONFIG);

    await user.click(screen.getByRole("switch"));
    expect(onChange).toHaveBeenNthCalledWith(2, false, null);
  });

  it("keeps every other key when the intent type changes", async () => {
    const user = userEvent.setup();
    render(<Harness />);

    await chooseOption(user, 0, "Generic (specify competitors manually)");

    expect(lastConfig()).toStrictEqual({ ...DEFAULT_CONFIG, competitor_intent_type: "generic" });
    expect(screen.getByText("Competitors")).toBeInTheDocument();
    expect(screen.queryByText("Locations (optional)")).not.toBeInTheDocument();
  });

  it("reports a policy change without dropping the other policy key", async () => {
    const user = userEvent.setup();
    render(<Harness />);

    await chooseOption(user, 3, "Reframe (suggest alternative)");

    expect(lastConfig()).toStrictEqual({
      ...DEFAULT_CONFIG,
      policy: { competitor_comparison: "reframe", possible_competitor_comparison: "reframe" },
    });
  });

  it("commits comma separated brand terms as separate tags", async () => {
    const user = userEvent.setup();
    render(<Harness />);

    const brandSelf = screen.getAllByRole("combobox")[1];
    await user.click(brandSelf);
    await user.type(brandSelf, "acme,globex,");

    expect(lastConfig().brand_self).toStrictEqual(["acme", "globex"]);
  });

  it("commits the pending brand term when the field loses focus", async () => {
    const user = userEvent.setup();
    render(<Harness />);

    const brandSelf = screen.getAllByRole("combobox")[1];
    await user.click(brandSelf);
    await user.type(brandSelf, "acme");
    await user.tab();

    expect(lastConfig().brand_self).toStrictEqual(["acme"]);
  });

  it("expands a picked airline into all of its match variants, lowercased", async () => {
    mockAirlines.mockResolvedValue({ airlines: [{ id: "qr", match: "Qatar Airways|qatar|qr", tags: [] }] });
    const user = userEvent.setup();
    render(<Harness />);

    await user.click(screen.getAllByRole("combobox")[1]);
    const options = await screen.findAllByText(/Qatar Airways/);
    await user.click(options[options.length - 1]);

    expect(lastConfig().brand_self).toStrictEqual(["qatar airways", "qatar", "qr"]);
  });

  it("reports locations only while the airline type is selected", async () => {
    const user = userEvent.setup();
    render(<Harness />);

    const locations = screen.getAllByRole("combobox")[2];
    await user.click(locations);
    await user.type(locations, "doha,");

    expect(lastConfig()).toStrictEqual({ ...DEFAULT_CONFIG, locations: ["doha"] });
  });

  it("reports a typed decimal threshold and leaves the other two alone", async () => {
    const user = userEvent.setup();
    render(<Harness />);

    const thresholds = screen.getAllByRole("spinbutton");
    await user.clear(thresholds[0]);
    await user.type(thresholds[0], "0.55");

    expect(lastConfig()).toStrictEqual({ ...DEFAULT_CONFIG, threshold_high: 0.55 });
  });

  it("falls back to the default threshold when the field is cleared", async () => {
    const user = userEvent.setup();
    render(<Harness />);

    await user.clear(screen.getAllByRole("spinbutton")[1]);

    expect(lastConfig()).toStrictEqual(DEFAULT_CONFIG);
  });

  it("clamps a threshold above the maximum back to 1 when the field is left", async () => {
    const user = userEvent.setup();
    render(<Harness />);

    const thresholds = screen.getAllByRole("spinbutton");
    await user.clear(thresholds[2]);
    await user.type(thresholds[2], "5");
    await user.tab();

    expect(lastConfig()).toStrictEqual({ ...DEFAULT_CONFIG, threshold_low: 1 });
  });

  it("explains the filter without rendering any control while switched off", () => {
    render(<Harness initialEnabled={false} />);

    expect(
      screen.getByText(
        "Block or reframe competitor comparison questions. When enabled, airline type auto-loads competitors from IATA; generic type requires manual competitor list.",
      ),
    ).toBeInTheDocument();
    expect(screen.queryAllByRole("combobox")).toHaveLength(0);
    expect(screen.queryAllByRole("spinbutton")).toHaveLength(0);
  });

  it.each([
    ["Type", "Airline (auto-load competitors from IATA)"],
    ["Policy: Competitor comparison", "Refuse (block request)"],
    ["Policy: Possible competitor comparison", "Reframe (suggest alternative to backend LLM)"],
  ])("shows the human label on the %s trigger", (name, label) => {
    render(<Harness />);

    expect(screen.getByRole("combobox", { name })).toHaveTextContent(label);
  });

  it("shows the human label on the Type trigger after switching to generic", async () => {
    const user = userEvent.setup();
    render(<Harness />);

    await user.click(screen.getByRole("combobox", { name: "Type" }));
    await user.click(await screen.findByRole("option", { name: "Generic (specify competitors manually)" }));

    expect(screen.getByRole("combobox", { name: "Type" })).toHaveTextContent("Generic (specify competitors manually)");
  });
});
