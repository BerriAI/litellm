const PRICES_SHOWN_ELSEWHERE = new Set(["input_cost_per_token", "output_cost_per_token"]);

const priceFormat = new Intl.NumberFormat("en-US", { maximumSignificantDigits: 6 });

const isPriceKey = (key: string) => key.includes("cost") && !PRICES_SHOWN_ELSEWHERE.has(key);

const isFinitePrice = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);

const isExtraPrice = (entry: [string, unknown]): entry is [string, number] =>
  isPriceKey(entry[0]) && isFinitePrice(entry[1]);

export function ModelExtraPrices({ model }: { model: Record<string, unknown> }) {
  const prices = Object.entries(model)
    .filter(isExtraPrice)
    .sort(([a], [b]) => a.localeCompare(b));
  if (prices.length === 0) {
    return null;
  }
  return (
    <div className="col-span-2">
      <p className="font-medium">Other Prices (USD):</p>
      <dl className="mt-1 grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1">
        {prices.map(([key, price]) => (
          <div key={key} className="contents">
            <dt className="text-muted-foreground">{key}</dt>
            <dd>${priceFormat.format(price)}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
