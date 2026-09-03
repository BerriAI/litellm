import { describe, expect, it } from "vitest";
import { DEFAULT_FORM_VALUES, formToPayload, parseCoreTools, storedValuesToForm } from "./toolSearchForm";

const STORED_SEMANTIC = {
  embedding_model: "text-embedding-3-small",
  top_k: 3,
  similarity_threshold: 0.25,
  core_tools: ["treasury-get_rates", "treasury-list_accounts"],
};

const SEMANTIC_FORM = {
  embedding_model: "text-embedding-3-small",
  top_k: 3,
  similarity_threshold: 0.25,
  core_tools_text: "treasury-get_rates\ntreasury-list_accounts",
};

const KEYWORD_PAYLOAD = { embedding_model: null, top_k: 5, similarity_threshold: 0, core_tools: [] };

const OVERSIZED_FORM = {
  embedding_model: "emb",
  top_k: 400,
  similarity_threshold: 0.5,
  core_tools_text: "treasury-get_rates",
};

const CLAMPED_PAYLOAD = {
  embedding_model: "emb",
  top_k: 100,
  similarity_threshold: 0.5,
  core_tools: ["treasury-get_rates"],
};

describe("storedValuesToForm", () => {
  it("maps stored settings onto the form, joining core tools one per line", () => {
    expect(storedValuesToForm(STORED_SEMANTIC)).toEqual(SEMANTIC_FORM);
  });

  it("falls back to keyword defaults when nothing usable is stored", () => {
    expect(storedValuesToForm({})).toEqual(DEFAULT_FORM_VALUES);
    expect(storedValuesToForm({ embedding_model: null, top_k: "7" })).toEqual(DEFAULT_FORM_VALUES);
  });
});

describe("parseCoreTools", () => {
  it("splits on newlines and commas, trims, drops blanks and duplicates in order", () => {
    expect(parseCoreTools(" a-x \n\nb-y, a-x ,c-z\n")).toEqual(["a-x", "b-y", "c-z"]);
  });
});

describe("formToPayload", () => {
  it("sends null for a cleared embedding model so the proxy returns to keyword matching", () => {
    expect(formToPayload({ ...DEFAULT_FORM_VALUES, embedding_model: "  " })).toEqual(KEYWORD_PAYLOAD);
  });

  it("clamps top_k into the range the proxy accepts and lists core tools", () => {
    expect(formToPayload(OVERSIZED_FORM)).toEqual(CLAMPED_PAYLOAD);
    expect(formToPayload({ ...DEFAULT_FORM_VALUES, top_k: 0 }).top_k).toBe(1);
  });
});
