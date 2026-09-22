import Foundation

struct GatewayClient: Sendable {
    let baseURL: URL

    static let defaultBaseURL = URL(string: "https://api.litellm.ai")!

    static func baseURL(from input: String) -> URL? {
        let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard var components = URLComponents(string: trimmed),
              let scheme = components.scheme?.lowercased(),
              ["http", "https"].contains(scheme),
              let host = components.host,
              !host.isEmpty,
              components.user == nil,
              components.password == nil,
              components.query == nil,
              components.fragment == nil else { return nil }
        while components.path.count > 1, components.path.hasSuffix("/") {
            components.path.removeLast()
        }
        return components.url
    }

    func login(username: String, password: String) async throws -> AuthSession {
        let url = baseURL.appending(path: "/v2/login")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: ["username": username, "password": password])
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else { throw GatewayError.invalidResponse }
        guard httpResponse.statusCode == 200 else {
            if httpResponse.statusCode == 401 { throw GatewayError.unauthorized }
            throw GatewayError.server(httpResponse.statusCode)
        }
        let payload = try JSONDecoder().decode(LoginResponse.self, from: data)
        guard let token = payload.token, let session = AuthSession(token: token) else {
            throw GatewayError.invalidLoginResponse
        }
        return session
    }

    func fetchUserInfo(apiKey: String) async throws -> UserInfoResponse {
        let url = baseURL.appending(path: "/v2/user/info")
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Accept")

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else { throw GatewayError.invalidResponse }
        switch httpResponse.statusCode {
        case 200...299: break
        case 401: throw GatewayError.unauthorized
        case 403: throw GatewayError.forbidden
        case 404: throw GatewayError.notFound
        default: throw GatewayError.server(httpResponse.statusCode)
        }

        return try decoder.decode(UserInfoResponse.self, from: data)
    }

    func fetchTeamInfo(apiKey: String, teamID: String) async throws -> TeamBudget {
        var components = URLComponents(url: baseURL.appending(path: "/team/info"), resolvingAgainstBaseURL: false)
        components?.queryItems = [URLQueryItem(name: "team_id", value: teamID)]
        guard let url = components?.url else { throw GatewayError.invalidResponse }
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Accept")

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else { throw GatewayError.invalidResponse }
        switch httpResponse.statusCode {
        case 200...299: break
        case 401: throw GatewayError.unauthorized
        case 403: throw GatewayError.forbidden
        case 404: throw GatewayError.notFound
        default: throw GatewayError.server(httpResponse.statusCode)
        }
        return try decoder.decode(TeamInfoResponse.self, from: data).teamInfo
    }

    func fetchKeyList(apiKey: String) async throws -> [String] {
        let url = baseURL.appending(path: "/key/list")
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Accept")

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else { throw GatewayError.invalidResponse }
        switch httpResponse.statusCode {
        case 200...299: break
        case 401: throw GatewayError.unauthorized
        case 403: throw GatewayError.forbidden
        case 404: throw GatewayError.notFound
        default: throw GatewayError.server(httpResponse.statusCode)
        }
        return try decoder.decode(KeyListResponse.self, from: data).keys
    }

    func fetchKeyInfo(apiKey: String, key: String) async throws -> KeyBudget {
        var components = URLComponents(url: baseURL.appending(path: "/key/info"), resolvingAgainstBaseURL: false)
        components?.queryItems = [URLQueryItem(name: "key", value: key)]
        guard let url = components?.url else { throw GatewayError.invalidResponse }
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Accept")

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else { throw GatewayError.invalidResponse }
        switch httpResponse.statusCode {
        case 200...299: break
        case 401: throw GatewayError.unauthorized
        case 403: throw GatewayError.forbidden
        case 404: throw GatewayError.notFound
        default: throw GatewayError.server(httpResponse.statusCode)
        }
        return try decoder.decode(KeyInfoResponse.self, from: data).info
    }

    private var decoder: JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .custom { decoder in
            let value = try decoder.singleValueContainer().decode(String.self)
            if let date = ISO8601DateFormatter().date(from: value) { return date }
            let formatter = DateFormatter()
            formatter.locale = Locale(identifier: "en_US_POSIX")
            formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXX"
            guard let date = formatter.date(from: value) else { throw GatewayError.invalidResponse }
            return date
        }
        return decoder
    }

}

private struct LoginResponse: Decodable {
    let token: String?
}
