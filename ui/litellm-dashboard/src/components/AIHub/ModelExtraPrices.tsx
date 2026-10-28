const PRICES_SHOWN_ELSEWHERE = new Set(["input_cost_per_token", "output_cost_per_token"]);

const TOKENS_PER_LISTED_PRICE = 1_000_000;

const priceFormat = new Intl.NumberFormat("en-US", { maximumSignificantDigits: 6 });

const isUsdPriceKey = (key: string) =>
  (key.includes("cost_per_") || key.includes("token_cost")) && !key.includes("_dbu_");

const isPerTokenPrice = (key: string) => /_token(_|$)/.test(key);

const isPriceKey = (key: string) => isUsdPriceKey(key) && !PRICES_SHOWN_ELSEWHERE.has(key);

const isFinitePrice = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);

const isExtraPrice = (entry: [string, unknown]): entry is [string, number] =>
  isPriceKey(entry[0]) && isFinitePrice(entry[1]);

const formatPrice = (key: string, price: number) =>
  isPerTokenPrice(key)
    ? `$${priceFormat.format(price * TOKENS_PER_LISTED_PRICE)} per 1M tokens`
    : `$${priceFormat.format(price)}`;

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
            <dd>{formatPrice(key, price)}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
