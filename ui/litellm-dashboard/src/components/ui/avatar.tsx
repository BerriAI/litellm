"use client";

import { Avatar as AvatarPrimitive } from "@base-ui/react/avatar";
import * as React from "react";

import { cn } from "@/lib/cva.config";

const Avatar = React.forwardRef<React.ComponentRef<typeof AvatarPrimitive.Root>, AvatarPrimitive.Root.Props>(
  ({ className, ...props }, ref) => (
    <AvatarPrimitive.Root
      ref={ref}
      data-slot="avatar"
      className={cn(
        "relative flex size-8 shrink-0 items-center justify-center overflow-hidden rounded-full",
        className,
      )}
      {...props}
    />
  ),
);
Avatar.displayName = "Avatar";

const AvatarImage = React.forwardRef<React.ComponentRef<typeof AvatarPrimitive.Image>, AvatarPrimitive.Image.Props>(
  ({ className, ...props }, ref) => (
    <AvatarPrimitive.Image
      ref={ref}
      data-slot="avatar-image"
      className={cn("size-full object-cover", className)}
      {...props}
    />
  ),
);
AvatarImage.displayName = "AvatarImage";

const AvatarFallback = React.forwardRef<
  React.ComponentRef<typeof AvatarPrimitive.Fallback>,
  AvatarPrimitive.Fallback.Props
>(({ className, ...props }, ref) => (
  <AvatarPrimitive.Fallback
    ref={ref}
    data-slot="avatar-fallback"
    className={cn("flex size-full items-center justify-center rounded-full text-xs font-medium", className)}
    {...props}
  />
));
AvatarFallback.displayName = "AvatarFallback";

export { Avatar, AvatarImage, AvatarFallback };
