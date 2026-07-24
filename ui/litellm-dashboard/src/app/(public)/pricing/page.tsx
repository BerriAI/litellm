import PricingTable from "@/components/public-relay/PricingTable";

export default function PricingPage() {
  return (
    <main className="mx-auto min-h-[72vh] max-w-7xl px-5 py-16 lg:px-8">
      <p className="font-mono text-xs uppercase tracking-[0.22em] text-[#d83a20]">Published rates</p>
      <h1 className="mt-5 text-5xl font-semibold tracking-[-0.045em]">Simple, fixed public pricing.</h1>
      <p className="mt-5 max-w-2xl text-lg leading-8 text-black/60">
        Prices are USD per million tokens. Each request keeps the price version active when it began.
      </p>
      <PricingTable />
    </main>
  );
}
