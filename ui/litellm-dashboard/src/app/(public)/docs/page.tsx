export default function DocsPage() {
  return (
    <main className="mx-auto min-h-[72vh] max-w-5xl px-5 py-16 lg:px-8">
      <p className="font-mono text-xs uppercase tracking-[0.22em] text-[#d83a20]">Quickstart</p>
      <h1 className="mt-5 text-5xl font-semibold tracking-[-0.045em]">Your first request.</h1>
      <div className="mt-12 grid gap-8 lg:grid-cols-[.65fr_1.35fr]">
        <ol className="space-y-5 text-sm leading-6 text-black/65">
          <li>1. Create an account and verify your email.</li>
          <li>2. Add prepaid balance with Stripe Checkout.</li>
          <li>3. Copy a relay key once and keep it server-side.</li>
          <li>4. Call models listed by GET /v1/models.</li>
        </ol>
        <pre className="overflow-x-auto rounded-3xl bg-[#171713] p-7 text-sm leading-7 text-white/80">
          <code>{`from openai import OpenAI

client = OpenAI(
    api_key=os.environ["RELAY_API_KEY"],
    base_url="https://api.example.com/v1",
)

response = client.responses.create(
    model="published-model",
    input="Hello from LiteLLM Relay",
)`}</code>
        </pre>
      </div>
    </main>
  );
}
