import { describe, expect, it } from "vitest";
import { validateWorkerAddress, analysisAccessSchema, workerFormSchema } from "./workerSchema";
import { workerSetupCommand, LENS_WORKER_IMAGE } from "./workerCommand";

const workerDefaults = {
  useExisting: false,
  analysisKey: null,
  access: { model: null, budget: "100" },
  address: "http://localhost:4000",
};

describe("worker setup", () => {
  it.each(["https://gateway.example/proxy", "http://host.docker.internal:4000"])("accepts %s", (address) => {
    expect(() => validateWorkerAddress(address)).not.toThrow();
  });

  it.each(["ftp://gateway.example", "https://user:pass@gateway.example", "https://user@gateway.example"])(
    "rejects %s",
    (address) => {
      expect(() => validateWorkerAddress(address)).toThrow("Enter an HTTP or HTTPS proxy URL without credentials");
    },
  );

  it("rejects malformed addresses", () => {
    expect(() => validateWorkerAddress("not a URL")).toThrow("Invalid URL");
  });

  it("quotes apostrophes literally and retains the pinned image and runtime restrictions", () => {
    const command = workerSetupCommand("https://gateway.example/proxy?name=it's", "token'quoted");
    expect(command).toContain("'LITELLM_URL=https://gateway.example/proxy?name=it'\\''s'");
    expect(command).toContain("'LENS_WORKER_TOKEN=token'\\''quoted'");
    expect(command).toContain("--read-only --cap-drop ALL");
    expect(command).toContain(
      "--security-opt no-new-privileges --platform linux/amd64 --add-host host.docker.internal:host-gateway",
    );
    expect(command.split("\n").at(-1)?.trim()).toBe(LENS_WORKER_IMAGE);
  });
});

it.each(["", "0", "-1", "NaN", "Infinity"])("rejects an invalid analysis budget: %s", (budget) => {
  const result = analysisAccessSchema.safeParse({ model: "analysis", budget });
  expect(result.success).toBe(false);
  if (!result.success)
    expect(result.error.issues[0].message).toBe("Choose a model and a monthly limit greater than zero");
});

it("requires a model and converts an accepted analysis budget to a number", () => {
  expect(analysisAccessSchema.safeParse({ model: null, budget: "100" }).success).toBe(false);
  expect(analysisAccessSchema.parse({ model: "analysis", budget: "0.01" })).toEqual({
    model: "analysis",
    budget: 0.01,
  });
});

it("validates the selected-key branch and non-empty proxy address with field paths", () => {
  const result = workerFormSchema.safeParse({
    ...workerDefaults,
    useExisting: true,
    address: " ",
  });
  expect(result.success).toBe(false);
  if (result.success) return;
  expect(result.error.issues.map(({ path }) => path)).toEqual([["analysisKey"], ["address"]]);
});

it("validates analysis access only when creating a new virtual key", () => {
  const invalidAccess = workerFormSchema.safeParse(workerDefaults);
  expect(invalidAccess.success).toBe(false);
  if (!invalidAccess.success) expect(invalidAccess.error.issues[0].path).toEqual(["access"]);
  expect(
    workerFormSchema.safeParse({
      ...workerDefaults,
      access: { model: "analysis", budget: "100" },
    }).success,
  ).toBe(true);
  expect(
    workerFormSchema.safeParse({
      ...workerDefaults,
      useExisting: true,
      analysisKey: "key",
    }).success,
  ).toBe(true);
});
