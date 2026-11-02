import ariaAgentImage from "../../../public/assets/aria-agent.png";
import { cn } from "@/lib/cva.config";

export function AriaAvatar({ size = 28, className }: { size?: number; className?: string }) {
  return (
    <img
      src={ariaAgentImage.src}
      width={size}
      height={size}
      alt=""
      aria-hidden="true"
      draggable={false}
      className={cn("shrink-0 object-contain [image-rendering:pixelated]", className)}
      style={{ width: size, height: size }}
    />
  );
}
