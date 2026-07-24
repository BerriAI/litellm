import StatusCard from "@/components/public-relay/StatusCard";

export default function StatusPage() {
  return (
    <main className="mx-auto min-h-[72vh] max-w-4xl px-5 py-16 lg:px-8">
      <p className="font-mono text-xs uppercase tracking-[0.22em] text-[#d83a20]">System status</p>
      <h1 className="mt-5 text-5xl font-semibold tracking-[-0.045em]">Relay availability.</h1>
      <StatusCard />
    </main>
  );
}
