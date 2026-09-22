import SwiftUI

@main
struct LiteLLMUsageMenuBarApp: App {
    @StateObject private var store = UsageStore()

    var body: some Scene {
        MenuBarExtra {
            UsagePopover().environmentObject(store)
        } label: {
            MenuBarLabel().environmentObject(store)
        }
        .menuBarExtraStyle(.window)

        Window("로그인 설정", id: "login-settings") {
            SettingsView().environmentObject(store)
        }
        .windowResizability(.contentSize)
        .defaultPosition(.center)
    }
}

private struct MenuBarLabel: View {
    @EnvironmentObject private var store: UsageStore

    var body: some View {
        switch store.state {
        case let .ready(snapshot): Text("🚅 \(snapshot.percentage.formatted(.number.precision(.fractionLength(0))))%")
        case .loading: Text("🚅 ...")
        case .needsKey: Text("🚅 --")
        case .unavailable, .failure: Text("🚅 !")
        }
    }
}
