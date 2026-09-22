// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "LiteLLMUsageMenuBar",
    platforms: [.macOS(.v13)],
    products: [.executable(name: "LiteLLMUsageMenuBar", targets: ["LiteLLMUsageMenuBar"])],
    targets: [.executableTarget(name: "LiteLLMUsageMenuBar")]
)
