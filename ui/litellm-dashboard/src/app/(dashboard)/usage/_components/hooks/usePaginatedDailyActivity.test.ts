import { describe, expect, it } from "vitest";
import { sumMetadata } from "./usePaginatedDailyActivity";

describe("sumMetadata", () => {
  it("sums flat cost across pages instead of keeping the first page's value", () => {
    // A team whose activity spans more than one page accrues flat cost on each of them.
    // Keeping page 1's value under-reports the Flat Cost and Total Cost tiles.
    const merged = sumMetadata({ total_spend: 1, total_flat_cost: 174.5 }, { total_spend: 2, total_flat_cost: 777 });

    expect(merged.total_flat_cost).toBe(951.5);
    expect(merged.total_spend).toBe(3);
  });

  it("treats a page missing the field as zero rather than dropping the running total", () => {
    expect(sumMetadata({ total_flat_cost: 480 }, {}).total_flat_cost).toBe(480);
    expect(sumMetadata({}, { total_flat_cost: 480 }).total_flat_cost).toBe(480);
  });

  it("carries non-summable keys through from the first page", () => {
    const merged = sumMetadata(
      { page: 1, total_pages: 3, total_spend: 1 },
      { page: 2, total_pages: 3, total_spend: 2 },
    );

    expect(merged.page).toBe(1);
    expect(merged.total_pages).toBe(3);
  });

  it("sums every total_* metric the daily activity metadata exposes", () => {
    // Guards the class of bug rather than one field: a new backend total that nobody adds
    // to SUMMABLE_METADATA_KEYS freezes at page 1, and spend still looks right so it reads
    // as trustworthy.
    const page = {
      total_spend: 1,
      total_prompt_tokens: 1,
      total_completion_tokens: 1,
      total_tokens: 1,
      total_api_requests: 1,
      total_successful_requests: 1,
      total_failed_requests: 1,
      total_cache_read_input_tokens: 1,
      total_cache_creation_input_tokens: 1,
      total_flat_cost: 1,
    };
    const merged = sumMetadata(page, page);

    for (const key of Object.keys(page)) {
      expect(merged[key], `${key} must be summed across pages`).toBe(2);
    }
  });
});
