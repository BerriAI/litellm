import Foundation
import Combine

@MainActor
final class UsageStore: ObservableObject {
    @Published private(set) var state: UsageState = .needsKey
    @Published private(set) var lastUpdated: Date?
    @Published private(set) var availableKeys: [GatewayKey] = []
    @Published private(set) var selectedKeyName: String?
    @Published private(set) var keyListMessage: String?
    @Published private(set) var isLoadingKeys = false
    @Published private(set) var autoRefreshInterval: AutoRefreshInterval
    private let keychain: KeychainStore
    private var client: GatewayClient
    private(set) var gatewayURL: URL
    private var credentials: (username: String, password: String)?
    private var session: AuthSession?
    private var isRefreshing = false
    private var refreshTimer: Timer?
    private var lastSpendSample: SpendSample?
    private var smoothedSpendPerDay: Double?
    private let selectedKeyDefaultsKey = "selected-key-name"
    private let refreshIntervalDefaultsKey = "auto-refresh-interval"
    private let gatewayURLDefaultsKey = "gateway-url"

    init(keychain: KeychainStore = KeychainStore(), client: GatewayClient? = nil) {
        let configuredURL = client?.baseURL ?? Self.loadGatewayURL()
        self.keychain = keychain
        self.client = client ?? GatewayClient(baseURL: configuredURL)
        gatewayURL = configuredURL
        selectedKeyName = UserDefaults.standard.string(forKey: selectedKeyDefaultsKey)
        let storedInterval = UserDefaults.standard.object(forKey: refreshIntervalDefaultsKey) as? Int
        autoRefreshInterval = AutoRefreshInterval(rawValue: storedInterval ?? AutoRefreshInterval.adaptive.rawValue) ?? .adaptive
        do {
            credentials = try keychain.readCredentials()
            state = hasCredentials ? .loading : .needsKey
        } catch {
            state = .failure(error.localizedDescription)
        }
        configureRefreshTimer()
        Task { [weak self] in
            await self?.loadKeys()
            await self?.refresh()
        }
    }

    var hasCredentials: Bool { credentials != nil }

    var savedUsername: String { credentials?.username ?? "" }
    var savedPassword: String { credentials?.password ?? "" }

    func saveSettings(gatewayURL input: String, username: String, password: String) throws {
        guard let newGatewayURL = GatewayClient.baseURL(from: input) else {
            throw GatewayError.invalidGatewayURL
        }
        let trimmedUsername = username.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedUsername.isEmpty, !password.isEmpty else { throw KeychainError(errSecParam) }
        try keychain.saveCredentials(username: trimmedUsername, password: password)
        UserDefaults.standard.set(newGatewayURL.absoluteString, forKey: gatewayURLDefaultsKey)
        gatewayURL = newGatewayURL
        client = GatewayClient(baseURL: newGatewayURL)
        credentials = (trimmedUsername, password)
        session = nil
        availableKeys = []
        keyListMessage = nil
        lastSpendSample = nil
        smoothedSpendPerDay = nil
        configureRefreshTimer()
        Task {
            while isRefreshing {
                try? await Task.sleep(for: .milliseconds(50))
            }
            await loadKeys()
            await refresh()
        }
    }

    func setAutoRefreshInterval(_ interval: AutoRefreshInterval) {
        guard autoRefreshInterval != interval else { return }
        autoRefreshInterval = interval
        UserDefaults.standard.set(interval.rawValue, forKey: refreshIntervalDefaultsKey)
        configureRefreshTimer()
    }

    func refresh() async {
        guard !isRefreshing, credentials != nil else { return }
        isRefreshing = true
        defer {
            isRefreshing = false
            scheduleNextRefresh()
        }
        do {
            let budget: TeamBudget
            do {
                let auth = try await authenticatedSession()
                budget = try await fetchCurrentBudget(apiKey: auth.apiKey)
            } catch GatewayError.unauthorized {
                session = nil
                let auth = try await authenticatedSession()
                budget = try await fetchCurrentBudget(apiKey: auth.apiKey)
            }
            guard let limit = budget.maxBudget, limit > 0 else { throw GatewayError.noUserBudget }
            guard let duration = budget.budgetDuration, isMonthly(duration) else {
                throw GatewayError.notMonthlyBudget(budget.budgetDuration ?? "없음")
            }
            let now = Date()
            updateSpendRate(spend: budget.spend, at: now)
            state = .ready(UsageSnapshot(
                spend: budget.spend,
                limit: limit,
                percentage: max(0, budget.spend / limit * 100),
                budgetDuration: duration,
                resetAt: budget.budgetResetAt
            ))
            lastUpdated = now
        } catch let error as GatewayError {
            state = .failure(error.localizedDescription)
        } catch {
            state = .failure("게이트웨이에 연결할 수 없습니다")
        }
    }

    func loadKeys() async {
        guard !isLoadingKeys, hasCredentials, availableKeys.isEmpty else { return }
        isLoadingKeys = true
        keyListMessage = nil
        defer { isLoadingKeys = false }
        do {
            let auth: AuthSession
            do {
                auth = try await authenticatedSession()
            } catch GatewayError.unauthorized {
                session = nil
                auth = try await authenticatedSession()
            }
            let keys = try await fetchKeyOptions(apiKey: auth.apiKey)
            availableKeys = keys
            if let selectedKeyName, !keys.contains(where: { $0.keyName == selectedKeyName }) {
                self.selectedKeyName = nil
                UserDefaults.standard.removeObject(forKey: selectedKeyDefaultsKey)
            }
        } catch let error as GatewayError {
            keyListMessage = error.localizedDescription
        } catch {
            keyListMessage = "키 목록을 불러올 수 없습니다"
        }
    }

    func selectKey(named keyName: String?) {
        guard selectedKeyName != keyName else { return }
        selectedKeyName = keyName
        lastSpendSample = nil
        smoothedSpendPerDay = nil
        if let keyName {
            UserDefaults.standard.set(keyName, forKey: selectedKeyDefaultsKey)
        } else {
            UserDefaults.standard.removeObject(forKey: selectedKeyDefaultsKey)
        }
        Task { await refresh() }
    }

    private func configureRefreshTimer() {
        refreshTimer?.invalidate()
        refreshTimer = nil
        scheduleNextRefresh()
    }

    private func scheduleNextRefresh() {
        refreshTimer?.invalidate()
        guard hasCredentials, let delay = nextRefreshDelay() else {
            refreshTimer = nil
            return
        }
        refreshTimer = Timer.scheduledTimer(withTimeInterval: delay, repeats: false) { [weak self] _ in
            Task { @MainActor [weak self] in
                await self?.refresh()
            }
        }
    }

    private func nextRefreshDelay() -> TimeInterval? {
        switch autoRefreshInterval {
        case .off:
            return nil
        case .adaptive:
            guard case let .ready(snapshot) = state else { return 900 }
            return adaptiveRefreshDelay(for: snapshot)
        case .oneMinute, .fiveMinutes, .fifteenMinutes, .thirtyMinutes, .oneHour:
            return autoRefreshInterval.fixedSeconds
        }
    }

    private func adaptiveRefreshDelay(for snapshot: UsageSnapshot) -> TimeInterval {
        if snapshot.percentage >= 100 {
            guard let resetAt = snapshot.resetAt else { return 86_400 }
            return min(86_400, max(60, resetAt.timeIntervalSinceNow - 60))
        }

        let remaining = max(0, snapshot.limit - snapshot.spend)
        if let smoothedSpendPerDay, smoothedSpendPerDay > 0 {
            let secondsUntilLimit = remaining / smoothedSpendPerDay * 86_400
            if secondsUntilLimit <= 6 * 3_600 { return 60 }
            if secondsUntilLimit <= 24 * 3_600 { return 300 }
            if secondsUntilLimit <= 3 * 86_400 { return 900 }
        }

        if snapshot.percentage >= 95 { return 300 }
        if snapshot.percentage >= 80 { return 900 }
        if snapshot.percentage >= 50 { return 3_600 }
        return 14_400
    }

    private func updateSpendRate(spend: Double, at date: Date) {
        if let lastSpendSample {
            let elapsed = date.timeIntervalSince(lastSpendSample.date)
            if elapsed > 0 {
                let spendDelta = spend - lastSpendSample.spend
                if spendDelta < 0 {
                    smoothedSpendPerDay = nil
                } else {
                    let spendPerDay = spendDelta / elapsed * 86_400
                    smoothedSpendPerDay = smoothedSpendPerDay.map { $0 * 0.7 + spendPerDay * 0.3 } ?? spendPerDay
                }
            }
        }
        lastSpendSample = SpendSample(spend: spend, date: date)
    }

    private struct SpendSample {
        let spend: Double
        let date: Date
    }

    private func authenticatedSession() async throws -> AuthSession {
        if let session, session.expiresAt.timeIntervalSinceNow > 300 { return session }
        guard let credentials else { throw GatewayError.unauthorized }
        let refreshed = try await client.login(username: credentials.username, password: credentials.password)
        session = refreshed
        return refreshed
    }

    private func fetchBudget(apiKey: String) async throws -> TeamBudget {
        let userInfo = try await client.fetchUserInfo(apiKey: apiKey)
        if userInfo.maxBudget != nil {
            return TeamBudget(
                spend: userInfo.spend,
                maxBudget: userInfo.maxBudget,
                budgetDuration: userInfo.budgetDuration,
                budgetResetAt: userInfo.budgetResetAt
            )
        }
        guard let teamID = userInfo.teams.first else { throw GatewayError.noUserBudget }
        return try await client.fetchTeamInfo(apiKey: apiKey, teamID: teamID)
    }

    private func fetchCurrentBudget(apiKey: String) async throws -> TeamBudget {
        guard let selectedKeyName else {
            return try await fetchBudget(apiKey: apiKey)
        }

        let keys = availableKeys.isEmpty ? try await fetchKeyOptions(apiKey: apiKey) : availableKeys
        availableKeys = keys
        guard let selectedKey = keys.first(where: { $0.keyName == selectedKeyName }) else {
            self.selectedKeyName = nil
            UserDefaults.standard.removeObject(forKey: selectedKeyDefaultsKey)
            return try await fetchBudget(apiKey: apiKey)
        }

        let keyInfo = try await client.fetchKeyInfo(apiKey: apiKey, key: selectedKey.value)
        if let maxBudget = keyInfo.maxBudget {
            return TeamBudget(
                spend: keyInfo.spend,
                maxBudget: maxBudget,
                budgetDuration: keyInfo.budgetDuration,
                budgetResetAt: keyInfo.budgetResetAt
            )
        }
        guard let teamID = keyInfo.teamID else { throw GatewayError.noUserBudget }
        let teamBudget = try await client.fetchTeamInfo(apiKey: apiKey, teamID: teamID)
        return TeamBudget(
            spend: keyInfo.spend,
            maxBudget: teamBudget.maxBudget,
            budgetDuration: teamBudget.budgetDuration,
            budgetResetAt: teamBudget.budgetResetAt
        )
    }

    private func fetchKeyOptions(apiKey: String) async throws -> [GatewayKey] {
        let keys = try await client.fetchKeyList(apiKey: apiKey)
        var options: [GatewayKey] = []
        for key in keys {
            let info = try await client.fetchKeyInfo(apiKey: apiKey, key: key)
            options.append(GatewayKey(value: key, keyName: info.keyName, alias: info.keyAlias))
        }
        return options
    }

    private func isMonthly(_ duration: String) -> Bool {
        let normalized = duration.lowercased()
        return ["1mo", "1month", "30d", "monthly"].contains(normalized)
    }

    private static func loadGatewayURL() -> URL {
        guard let savedURL = UserDefaults.standard.string(forKey: "gateway-url"),
              let gatewayURL = GatewayClient.baseURL(from: savedURL) else {
            return GatewayClient.defaultBaseURL
        }
        return gatewayURL
    }

}
