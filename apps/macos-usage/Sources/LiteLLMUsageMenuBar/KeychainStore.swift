import Foundation
import Security
import LocalAuthentication

struct KeychainStore: Sendable {
    private let service = "com.litellm.usage-menubar"
    private let credentialsAccount = "ui-credentials"

    func readCredentials() throws -> (username: String, password: String)? {
        guard let value = try read(account: credentialsAccount) else { return nil }
        return try JSONDecoder().decode(StoredCredentials.self, from: Data(value.utf8)).tuple
    }

    func saveCredentials(username: String, password: String) throws {
        let value = try JSONEncoder().encode(StoredCredentials(username: username, password: password))
        try save(String(data: value, encoding: .utf8) ?? "", account: credentialsAccount)
    }

    private func read(account: String) throws -> String? {
        let context = LAContext()
        context.interactionNotAllowed = true
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
            kSecUseAuthenticationContext as String: context
        ]
        var result: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        if status == errSecItemNotFound { return nil }
        guard status == errSecSuccess, let data = result as? Data else {
            throw KeychainError(status)
        }
        return String(data: data, encoding: .utf8)
    }

    private func save(_ value: String, account: String) throws {
        let data = Data(value.utf8)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account
        ]
        var item = query
        item[kSecValueData as String] = data
        let addStatus = SecItemAdd(item as CFDictionary, nil)
        if addStatus == errSecSuccess { return }
        guard addStatus == errSecDuplicateItem else {
            throw KeychainError(addStatus)
        }
        let updateStatus = SecItemUpdate(query as CFDictionary, [kSecValueData as String: data] as CFDictionary)
        if updateStatus != errSecSuccess {
            throw KeychainError(updateStatus)
        }
    }
}

private struct StoredCredentials: Codable {
    let username: String
    let password: String

    var tuple: (username: String, password: String) {
        (username, password)
    }
}

struct KeychainError: LocalizedError, Equatable {
    let status: OSStatus

    init(_ status: OSStatus) { self.status = status }

    var errorDescription: String? {
        if status == errSecInteractionNotAllowed || status == errSecAuthFailed || status == errSecUserCanceled {
            return "저장된 로그인 정보에 접근할 수 없습니다. 설정에서 다시 저장하세요. Keychain 승인은 저장할 때만 요청합니다"
        }
        return "Keychain 처리에 실패했습니다 (\(status))"
    }
}
