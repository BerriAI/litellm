param imageName string = 'ghcr.io/berriai/litellm:main-latest'
param containerName string = 'litellm-container'
param dnsLabelName string = 'litellm'
param portNumber int = 4000

resource containerGroupName 'Microsoft.ContainerInstance/containerGroups@2021-03-01' = {
  name: containerName
  location: resourceGroup().location
  properties: {
    containers: [
      {
        name: containerName
        properties: {
          image: imageName
          resources: {
            requests: {
              cpu: 1
              memoryInGB: 2
            }
          }
          ports: [
            {
              port: portNumber
            }
          ]
        }
      }
    ]
    osType: 'Linux'
    restartPolicy: 'Always'
    ipAddress: {
      type: 'Public'
      ports: [
        {
          protocol: 'tcp'
          port: portNumber
        }
      ]
      dnsNameLabel: dnsLabelName
    }
  }
}
