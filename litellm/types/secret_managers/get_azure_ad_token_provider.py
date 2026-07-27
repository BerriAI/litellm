from enum import Enum


class AzureCredentialType(str, Enum):
    ClientSecretCredential = "ClientSecretCredential"
    ManagedIdentityCredential = "ManagedIdentityCredential"
    CertificateCredential = "CertificateCredential"
    WorkloadIdentityCredential = "WorkloadIdentityCredential"
    DefaultAzureCredential = "DefaultAzureCredential"
