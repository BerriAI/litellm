import SwiftUI

struct UsagePopover: View {
    @EnvironmentObject private var store: UsageStore
    @Environment(\.openWindow) private var openWindow
    private let teamSelection = "__team__"

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            scopePicker
            content
            Divider()
            actions
        }
        .padding(16)
        .frame(width: 320)
        .task {
            await store.loadKeys()
            await store.refresh()
        }
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("LiteLLM 사용량").font(.headline)
            }
            Spacer()
            statusIcon
        }
    }

    private var scopePicker: some View {
        VStack(alignment: .leading, spacing: 4) {
            Picker("조회 대상", selection: Binding(
                get: { store.selectedKeyName ?? teamSelection },
                set: { store.selectKey(named: $0 == teamSelection ? nil : $0) }
            )) {
                Text("팀 전체").tag(teamSelection)
                ForEach(store.availableKeys) { key in
                    Text(key.displayName).tag(key.keyName)
                }
            }
            .pickerStyle(.menu)
            .disabled(!store.hasCredentials || store.isLoadingKeys)
            if let keyListMessage = store.keyListMessage, !store.isLoadingKeys {
                Text(keyListMessage).font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder
    private var content: some View {
        switch store.state {
        case .needsKey:
            MessageView(title: "로그인 정보를 설정하세요", detail: "설정에서 게이트웨이 ID와 비밀번호를 입력하면 사용량을 조회합니다", symbol: "person.crop.circle.badge.key")
        case .loading:
            HStack(spacing: 8) { ProgressView(); Text("사용량을 확인하는 중...").foregroundStyle(.secondary) }
        case let .ready(snapshot):
            VStack(alignment: .leading, spacing: 10) {
                Text(snapshot.percentage.formatted(.number.precision(.fractionLength(0))) + "%")
                    .font(.system(size: 38, weight: .bold, design: .rounded))
                    .foregroundStyle(color(for: snapshot.percentage))
                ProgressView(value: min(snapshot.percentage, 100), total: 100)
                    .tint(color(for: snapshot.percentage))
                HStack {
                    Fact(label: "사용액", value: currency(snapshot.spend))
                    Spacer()
                    Fact(label: "한도", value: currency(snapshot.limit))
                }
                if let resetAt = snapshot.resetAt {
                    Text("다음 초기화 " + resetAt.formatted(date: .abbreviated, time: .shortened))
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
        case let .unavailable(message), let .failure(message):
            MessageView(title: "사용량을 표시할 수 없습니다", detail: message, symbol: "exclamationmark.triangle")
        }
    }

    private var actions: some View {
        HStack {
            Button("새로고침") { Task { await store.refresh() } }
                .buttonStyle(.borderless)
                .disabled(store.state == .loading || !store.hasCredentials)
            Spacer()
            Button("설정") {
                openWindow(id: "login-settings")
                NSApplication.shared.activate(ignoringOtherApps: true)
            }
                .buttonStyle(.borderless)
            Button("종료") { NSApplication.shared.terminate(nil) }
                .buttonStyle(.borderless)
        }
        .font(.callout)
    }

    @ViewBuilder
    private var statusIcon: some View {
        switch store.state {
        case .ready(let snapshot): Image(systemName: "circle.fill").foregroundStyle(color(for: snapshot.percentage))
        case .loading: ProgressView().controlSize(.small)
        case .needsKey: Image(systemName: "key.fill").foregroundStyle(.secondary)
        case .unavailable, .failure: Image(systemName: "exclamationmark.circle.fill").foregroundStyle(.red)
        }
    }

    private func color(for percentage: Double) -> Color {
        percentage >= 90 ? .red : percentage >= 75 ? .orange : .green
    }

    private func currency(_ value: Double) -> String { value.formatted(.currency(code: "USD")) }
}

struct SettingsView: View {
    @EnvironmentObject private var store: UsageStore
    @Environment(\.dismiss) private var dismiss
    @State private var gatewayURL = ""
    @State private var username = ""
    @State private var password = ""
    @State private var message: String?

    var body: some View {
        Form {
            Section("사용량") {
                Picker("자동 새로고침", selection: Binding(
                    get: { store.autoRefreshInterval },
                    set: { store.setAutoRefreshInterval($0) }
                )) {
                    ForEach(AutoRefreshInterval.allCases) { interval in
                        Text(interval.title).tag(interval)
                    }
                }
                Text("팝오버가 닫혀 있어도 메뉴바 사용량을 갱신합니다")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("게이트웨이") {
                TextField("서버 주소", text: $gatewayURL)
                TextField("아이디", text: $username)
                SecureField("비밀번호", text: $password)
                Text("LiteLLM 프록시 서버 주소를 입력하세요")
                    .font(.caption).foregroundStyle(.secondary)
                Text("로그인 정보는 macOS Keychain에 저장되며 세션은 앱 실행 중 메모리에만 유지됩니다")
                    .font(.caption).foregroundStyle(.secondary)
                Button("저장") {
                    guard !username.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, !password.isEmpty else {
                        message = "아이디와 비밀번호를 모두 입력하세요"
                        return
                    }
                    do {
                        try store.saveSettings(gatewayURL: gatewayURL, username: username, password: password)
                        username = ""
                        password = ""
                        dismiss()
                    } catch { message = error.localizedDescription }
                }
                .disabled(gatewayURL.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || username.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || password.isEmpty)
                if let message { Text(message).font(.caption).foregroundStyle(.secondary) }
            }
        }
        .formStyle(.grouped)
        .frame(width: 380, height: 360)
        .padding()
        .onAppear {
            gatewayURL = store.gatewayURL.absoluteString
            username = store.savedUsername
            password = store.savedPassword
        }
    }
}

private struct MessageView: View {
    let title: String
    let detail: String
    let symbol: String

    var body: some View {
        Label {
            VStack(alignment: .leading, spacing: 3) {
                Text(title).font(.subheadline.weight(.semibold))
                Text(detail).font(.caption).foregroundStyle(.secondary)
            }
        } icon: { Image(systemName: symbol).foregroundStyle(.secondary) }
    }
}

private struct Fact: View {
    let label: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.caption).foregroundStyle(.secondary)
            Text(value).font(.system(.callout, design: .monospaced))
        }
    }
}
