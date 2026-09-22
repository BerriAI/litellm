import Foundation

struct UserInfoResponse: Decodable, Sendable {
    let userID: String
    let spend: Double
    let maxBudget: Double?
    let budgetDuration: String?
    let budgetResetAt: Date?
    let teams: [String]

    enum CodingKeys: String, CodingKey {
        case userID = "user_id"
        case spend
        case maxBudget = "max_budget"
        case budgetDuration = "budget_duration"
        case budgetResetAt = "budget_reset_at"
        case teams
    }
}

struct TeamInfoResponse: Decodable, Sendable {
    let teamInfo: TeamBudget

    enum CodingKeys: String, CodingKey {
        case teamInfo = "team_info"
    }
}

struct TeamBudget: Decodable, Sendable {
    let spend: Double
    let maxBudget: Double?
    let budgetDuration: String?
    let budgetResetAt: Date?

    enum CodingKeys: String, CodingKey {
        case spend
        case maxBudget = "max_budget"
        case budgetDuration = "budget_duration"
        case budgetResetAt = "budget_reset_at"
    }
}

struct KeyListResponse: Decodable, Sendable {
    let keys: [String]
}

struct KeyInfoResponse: Decodable, Sendable {
    let info: KeyBudget

    enum CodingKeys: String, CodingKey {
        case info
    }
}

struct KeyBudget: Decodable, Sendable {
    let keyAlias: String?
    let keyName: String
    let spend: Double
    let maxBudget: Double?
    let budgetDuration: String?
    let budgetResetAt: Date?
    let teamID: String?

    enum CodingKeys: String, CodingKey {
        case keyAlias = "key_alias"
        case keyName = "key_name"
        case spend
        case maxBudget = "max_budget"
        case budgetDuration = "budget_duration"
        case budgetResetAt = "budget_reset_at"
        case teamID = "team_id"
    }
}

struct GatewayKey: Identifiable, Equatable, Sendable {
    let value: String
    let keyName: String
    let alias: String?

    var id: String { keyName }
    var displayName: String { alias ?? keyName }
}

struct AuthSession: Sendable {
    let token: String
    let apiKey: String
    let expiresAt: Date

    init?(token: String) {
        let segments: [Substring] = token.split(separator: ".")
        guard segments.count == 3 else { return nil }
        var encoded = String(segments[1]).replacingOccurrences(of: "-", with: "+").replacingOccurrences(of: "_", with: "/")
        encoded += String(repeating: "=", count: (4 - encoded.count % 4) % 4)
        guard let data = Data(base64Encoded: encoded),
              let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let apiKey = payload["key"] as? String,
              let exp = payload["exp"] as? TimeInterval else { return nil }
        self.token = token
        self.apiKey = apiKey
        self.expiresAt = Date(timeIntervalSince1970: exp)
    }
}

enum UsageState: Equatable, Sendable {
    case needsKey
    case loading
    case ready(UsageSnapshot)
    case unavailable(String)
    case failure(String)
}

enum AutoRefreshInterval: Int, CaseIterable, Identifiable, Sendable {
    case off = 0
    case adaptive = -1
    case oneMinute = 60
    case fiveMinutes = 300
    case fifteenMinutes = 900
    case thirtyMinutes = 1800
    case oneHour = 3600

    var id: Self { self }

    var fixedSeconds: TimeInterval? {
        switch self {
        case .off, .adaptive: nil
        case .oneMinute, .fiveMinutes, .fifteenMinutes, .thirtyMinutes, .oneHour:
            TimeInterval(rawValue)
        }
    }

    var title: String {
        switch self {
        case .off: "끔"
        case .adaptive: "적응형"
        case .oneMinute: "1분"
        case .fiveMinutes: "5분"
        case .fifteenMinutes: "15분"
        case .thirtyMinutes: "30분"
        case .oneHour: "1시간"
        }
    }
}

struct UsageSnapshot: Equatable, Sendable {
    let spend: Double
    let limit: Double
    let percentage: Double
    let budgetDuration: String
    let resetAt: Date?
}

enum GatewayError: LocalizedError {
    case invalidResponse
    case unauthorized
    case forbidden
    case notFound
    case server(Int)
    case noUserBudget
    case notMonthlyBudget(String)
    case invalidLoginResponse
    case invalidGatewayURL

    var errorDescription: String? {
        switch self {
        case .invalidResponse: "게이트웨이 응답을 읽을 수 없습니다"
        case .unauthorized: "로그인 정보가 올바르지 않거나 세션이 만료되었습니다"
        case .forbidden: "개인 사용량 조회가 거부되었습니다 (HTTP 403)"
        case .notFound: "사용자를 찾을 수 없습니다"
        case let .server(code): "게이트웨이 오류 (HTTP \(code))"
        case .noUserBudget: "개인 예산 한도가 설정되지 않았습니다"
        case let .notMonthlyBudget(duration): "개인 예산이 월 단위가 아닙니다 (\(duration))"
        case .invalidLoginResponse: "로그인 응답을 읽을 수 없습니다"
        case .invalidGatewayURL: "올바른 게이트웨이 URL을 입력하세요 (예: https://gateway.example.com)"
        }
    }
}
