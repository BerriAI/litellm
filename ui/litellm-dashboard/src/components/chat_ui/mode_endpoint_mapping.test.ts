import { describe, expect, it } from "vitest";
import { EndpointType, getEndpointType, litellmModeMapping, ModelMode } from "./mode_endpoint_mapping";

describe("mode_endpoint_mapping", () => {
  it("maps OCR model mode to the OCR endpoint", () => {
    expect(getEndpointType("ocr")).toBe(EndpointType.OCR);
    expect(litellmModeMapping[ModelMode.OCR]).toBe(EndpointType.OCR);
  });

  it("preserves existing model mode mappings", () => {
    expect(getEndpointType("chat")).toBe(EndpointType.CHAT);
    expect(getEndpointType("responses")).toBe(EndpointType.RESPONSES);
    expect(getEndpointType("image_generation")).toBe(EndpointType.IMAGE);
    expect(getEndpointType("audio_transcription")).toBe(EndpointType.TRANSCRIPTION);
  });
});
