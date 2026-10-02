export function ActiveDot() {
  return (
    <span className="relative flex size-2" aria-hidden="true">
      <span className="absolute inline-flex size-full animate-ping rounded-full bg-emerald-500 opacity-60 motion-reduce:hidden" />
      <span className="relative inline-flex size-2 rounded-full bg-emerald-500" />
    </span>
  );
}
