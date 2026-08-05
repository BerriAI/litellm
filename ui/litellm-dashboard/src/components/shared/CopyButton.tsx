import { Button } from "@/components/ui/button";
import { cn } from "@/lib/cva.config";
import { Check, Copy } from "lucide-react";
import React, { useEffect, useState } from "react";

interface CopyButtonProps {
  value: string | null | undefined;
  label: string;
  className?: string;
  iconClassName?: string;
}

const CopyButton: React.FC<CopyButtonProps> = ({ value, label, className, iconClassName = "size-[15px]" }) => {
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!copied) return;
    const timer = setTimeout(() => setCopied(false), 1200);
    return () => clearTimeout(timer);
  }, [copied]);

  if (!value) return null;

  const handleCopy = async () => {
    if (!navigator.clipboard) return;
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  };

  return (
    <Button
      type="button"
      variant="ghost"
      size="icon-xs"
      onClick={handleCopy}
      aria-label={label}
      title={label}
      className={cn("text-muted-foreground hover:text-primary", className)}
    >
      {copied ? <Check className={iconClassName} /> : <Copy className={iconClassName} />}
    </Button>
  );
};

export default CopyButton;
