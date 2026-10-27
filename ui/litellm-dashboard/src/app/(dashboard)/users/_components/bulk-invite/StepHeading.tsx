export function StepHeading({ step, title }: { step: number; title: string }) {
  return (
    <div className="flex items-center mb-4">
      <div className="w-8 h-8 rounded-full bg-info text-info-foreground flex items-center justify-center mr-3">
        {step}
      </div>
      <h3 className="text-lg font-medium">{title}</h3>
    </div>
  );
}
