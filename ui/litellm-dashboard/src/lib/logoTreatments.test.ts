import { describe, expect, it } from "vitest";
import { logoTreatmentFor } from "./logoTreatments";

describe("logoTreatmentFor", () => {
  it("marks a monochrome transparent mark for inversion", () => {
    expect(logoTreatmentFor("/ui/assets/logos/github.svg")).toBe("invert");
  });

  it("marks a multicolor dark mark for a plate instead of inversion", () => {
    expect(logoTreatmentFor("/ui/assets/logos/fireworks.svg")).toBe("plate");
  });

  it("plates a dark mark with a light knockout, which inversion would flatten away", () => {
    expect(logoTreatmentFor("/ui/assets/logos/repelloai.png")).toBe("plate");
    expect(logoTreatmentFor("/ui/assets/logos/aiml_api.svg")).toBe("plate");
  });

  it("leaves an asset that already reads on dark untreated", () => {
    expect(logoTreatmentFor("/ui/assets/logos/slack.svg")).toBeUndefined();
  });

  it("resolves through a bundler fingerprint in the filename", () => {
    expect(logoTreatmentFor("/litellm-asset-prefix/_next/static/media/openrouter.1xk7748-_jixf.svg")).toBe("invert");
  });

  it("resolves a bundled asset served under a proxy root path", () => {
    expect(logoTreatmentFor("/litellm/ui/assets/logos/notion.svg")).toBe("invert");
  });

  it("ignores a query string and fragment on the asset URL", () => {
    expect(logoTreatmentFor("/ui/assets/logos/vercel.svg?v=2#icon")).toBe("invert");
  });

  it("does not treat an external URL whose filename collides with a bundled asset", () => {
    expect(logoTreatmentFor("https://cdn.example.com/github.svg")).toBeUndefined();
  });

  it("does not treat a non-logo path whose filename collides with a bundled asset", () => {
    expect(logoTreatmentFor("/uploads/user/github.svg")).toBeUndefined();
  });

  it("leaves an opaque dark box untreated, since a plate behind it cannot show through", () => {
    expect(logoTreatmentFor("/ui/assets/logos/lakeraai.jpeg")).toBeUndefined();
  });

  it("returns undefined for empty and nullish input", () => {
    expect(logoTreatmentFor(null)).toBeUndefined();
    expect(logoTreatmentFor(undefined)).toBeUndefined();
    expect(logoTreatmentFor("")).toBeUndefined();
  });

  it("distinguishes assets that share a stem but differ by extension", () => {
    expect(logoTreatmentFor("/ui/assets/logos/runway.png")).toBe("invert");
    expect(logoTreatmentFor("/ui/assets/logos/runway.svg")).toBeUndefined();
  });
});
