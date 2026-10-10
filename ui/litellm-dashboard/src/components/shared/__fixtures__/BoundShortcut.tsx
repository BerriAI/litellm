import { type ShortcutLayer, useShortcut } from "../useShortcut";

/** Binds one shortcut and reports which binding fired. */
export function BoundShortcut({
  id,
  keys,
  layer,
  enabled = true,
  onPress,
}: {
  id: string;
  keys: readonly [string, string];
  layer: ShortcutLayer;
  enabled?: boolean;
  onPress: (id: string) => void;
}) {
  useShortcut(keys[0], () => onPress(id), { layer, enabled, description: keys[1] });
  return null;
}
