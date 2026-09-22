#!/bin/sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
BUILD_DIR="$ROOT/.build/release"
APP_DIR="$ROOT/LiteLLMUsage.app"
DMG_PATH="$ROOT/LiteLLMUsage.dmg"
ICON_SOURCE="$ROOT/assets/litellm_logo.jpg"
ICON_RENDER_DIR="$ROOT/.build/icon-render"
ICONSET_DIR="$ROOT/.build/LiteLLMUsage.iconset"

swift build -c release --package-path "$ROOT"
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS" "$APP_DIR/Contents/Resources"
cp "$BUILD_DIR/LiteLLMUsageMenuBar" "$APP_DIR/Contents/MacOS/LiteLLMUsageMenuBar"
rm -rf "$ICON_RENDER_DIR" "$ICONSET_DIR"
mkdir -p "$ICON_RENDER_DIR" "$ICONSET_DIR"
swift "$ROOT/scripts/fill-logo-background.swift" "$ICON_SOURCE" "$ICON_RENDER_DIR/logo-mark.png"
ICON_PNG="$ICON_RENDER_DIR/logo-mark.png"
sips -z 16 16 "$ICON_PNG" --out "$ICONSET_DIR/icon_16x16.png" >/dev/null
sips -z 32 32 "$ICON_PNG" --out "$ICONSET_DIR/icon_16x16@2x.png" >/dev/null
sips -z 32 32 "$ICON_PNG" --out "$ICONSET_DIR/icon_32x32.png" >/dev/null
sips -z 64 64 "$ICON_PNG" --out "$ICONSET_DIR/icon_32x32@2x.png" >/dev/null
sips -z 128 128 "$ICON_PNG" --out "$ICONSET_DIR/icon_128x128.png" >/dev/null
sips -z 256 256 "$ICON_PNG" --out "$ICONSET_DIR/icon_128x128@2x.png" >/dev/null
sips -z 256 256 "$ICON_PNG" --out "$ICONSET_DIR/icon_256x256.png" >/dev/null
sips -z 512 512 "$ICON_PNG" --out "$ICONSET_DIR/icon_256x256@2x.png" >/dev/null
sips -z 512 512 "$ICON_PNG" --out "$ICONSET_DIR/icon_512x512.png" >/dev/null
sips -z 1024 1024 "$ICON_PNG" --out "$ICONSET_DIR/icon_512x512@2x.png" >/dev/null
iconutil -c icns "$ICONSET_DIR" -o "$APP_DIR/Contents/Resources/LiteLLMUsage.icns"
cat > "$APP_DIR/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleDisplayName</key><string>LiteLLM Usage</string>
<key>CFBundleExecutable</key><string>LiteLLMUsageMenuBar</string>
<key>CFBundleIdentifier</key><string>com.litellm.usage-menubar</string>
<key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
<key>CFBundleName</key><string>LiteLLM Usage</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleShortVersionString</key><string>0.1.0</string>
<key>CFBundleVersion</key><string>1</string>
<key>CFBundleIconFile</key><string>LiteLLMUsage.icns</string>
<key>LSMinimumSystemVersion</key><string>13.0</string>
<key>LSUIElement</key><true/>
</dict></plist>
PLIST
SIGNING_IDENTITY="${CODESIGN_IDENTITY:-LiteLLM Usage Development}"
if ! security find-identity -v -p codesigning | grep -Fq "\"$SIGNING_IDENTITY\""; then
    printf 'Missing valid codesigning identity: %s\n' "$SIGNING_IDENTITY" >&2
    exit 1
fi
codesign --force --deep --timestamp=none --sign "$SIGNING_IDENTITY" "$APP_DIR" >/dev/null
rm -f "$DMG_PATH"
hdiutil create -volname "LiteLLM Usage" -srcfolder "$APP_DIR" -ov -format UDZO "$DMG_PATH" >/dev/null
printf '%s\n%s\n' "$APP_DIR" "$DMG_PATH"
