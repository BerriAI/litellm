# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


K8S_CLUSTER_NAME = "k8s.cluster.name"
"""
The name of the cluster.
"""

K8S_CLUSTER_UID = "k8s.cluster.uid"
"""
A pseudo-ID for the cluster, set to the UID of the `kube-system` namespace.
Note: K8s doesn't have support for obtaining a cluster ID. If this is ever
    added, we will recommend collecting the `k8s.cluster.uid` through the
    official APIs. In the meantime, we are able to use the `uid` of the
    `kube-system` namespace as a proxy for cluster ID. Read on for the
    rationale.

    Every object created in a K8s cluster is assigned a distinct UID. The
    `kube-system` namespace is used by Kubernetes itself and will exist
    for the lifetime of the cluster. Using the `uid` of the `kube-system`
    namespace is a reasonable proxy for the K8s ClusterID as it will only
    change if the cluster is rebuilt. Furthermore, Kubernetes UIDs are
    UUIDs as standardized by
    [ISO/IEC 9834-8 and ITU-T X.667](https://www.itu.int/ITU-T/studygroups/com17/oid.html).
    Which states:

    > If generated according to one of the mechanisms defined in Rec.
      ITU-T X.667 | ISO/IEC 9834-8, a UUID is either guaranteed to be
      different from all other UUIDs generated before 3603 A.D., or is
      extremely likely to be different (depending on the mechanism chosen).

    Therefore, UIDs between clusters should be extremely unlikely to
    conflict.
"""

K8S_CONTAINER_NAME = "k8s.container.name"
"""
The name of the Container from Pod specification, must be unique within a Pod. Container runtime usually uses different globally unique name (`container.name`).
"""

K8S_CONTAINER_RESTART_COUNT = "k8s.container.restart_count"
"""
Number of times the container was restarted. This attribute can be used to identify a particular container (running or stopped) within a container spec.
"""

K8S_CRONJOB_NAME = "k8s.cronjob.name"
"""
The name of the CronJob.
"""

K8S_CRONJOB_UID = "k8s.cronjob.uid"
"""
The UID of the CronJob.
"""

K8S_DAEMONSET_NAME = "k8s.daemonset.name"
"""
The name of the DaemonSet.
"""

K8S_DAEMONSET_UID = "k8s.daemonset.uid"
"""
The UID of the DaemonSet.
"""

K8S_DEPLOYMENT_NAME = "k8s.deployment.name"
"""
The name of the Deployment.
"""

K8S_DEPLOYMENT_UID = "k8s.deployment.uid"
"""
The UID of the Deployment.
"""

K8S_JOB_NAME = "k8s.job.name"
"""
The name of the Job.
"""

K8S_JOB_UID = "k8s.job.uid"
"""
The UID of the Job.
"""

K8S_NAMESPACE_NAME = "k8s.namespace.name"
"""
The name of the namespace that the pod is running in.
"""

K8S_NODE_NAME = "k8s.node.name"
"""
The name of the Node.
"""

K8S_NODE_UID = "k8s.node.uid"
"""
The UID of the Node.
"""

K8S_POD_ANNOTATION_TEMPLATE = "k8s.pod.annotation"
"""
The annotation key-value pairs placed on the Pod, the `<key>` being the annotation name, the value being the annotation value.
"""

K8S_POD_LABEL_TEMPLATE = "k8s.pod.label"
"""
The label key-value pairs placed on the Pod, the `<key>` being the label name, the value being the label value.
"""

K8S_POD_LABELS_TEMPLATE = "k8s.pod.labels"
"""
Deprecated: Replaced by `k8s.pod.label`.
"""

K8S_POD_NAME = "k8s.pod.name"
"""
The name of the Pod.
"""

K8S_POD_UID = "k8s.pod.uid"
"""
The UID of the Pod.
"""

K8S_REPLICASET_NAME = "k8s.replicaset.name"
"""
The name of the ReplicaSet.
"""

K8S_REPLICASET_UID = "k8s.replicaset.uid"
"""
The UID of the ReplicaSet.
"""

K8S_STATEFULSET_NAME = "k8s.statefulset.name"
"""
The name of the StatefulSet.
"""

K8S_STATEFULSET_UID = "k8s.statefulset.uid"
"""
The UID of the StatefulSet.
"""
