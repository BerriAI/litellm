from enum import Enum


class AzureCredentialType(str, Enum):
    ClientSecretCredential = "ClientSecretCredential"
    ManagedIdentityCredential = "ManagedIdentityCredential"
    CertificateCredential = "CertificateCredential"
