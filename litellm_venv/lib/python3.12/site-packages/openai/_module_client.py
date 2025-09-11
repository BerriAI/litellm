# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

if TYPE_CHECKING:
    from .resources.files import Files
    from .resources.images import Images
    from .resources.models import Models
    from .resources.batches import Batches
    from .resources.webhooks import Webhooks
    from .resources.beta.beta import Beta
    from .resources.chat.chat import Chat
    from .resources.embeddings import Embeddings
    from .resources.audio.audio import Audio
    from .resources.completions import Completions
    from .resources.evals.evals import Evals
    from .resources.moderations import Moderations
    from .resources.uploads.uploads import Uploads
    from .resources.responses.responses import Responses
    from .resources.containers.containers import Containers
    from .resources.fine_tuning.fine_tuning import FineTuning
    from .resources.vector_stores.vector_stores import VectorStores

from . import _load_client
from ._utils import LazyProxy


class ChatProxy(LazyProxy["Chat"]):
    @override
    def __load__(self) -> Chat:
        return _load_client().chat


class BetaProxy(LazyProxy["Beta"]):
    @override
    def __load__(self) -> Beta:
        return _load_client().beta


class FilesProxy(LazyProxy["Files"]):
    @override
    def __load__(self) -> Files:
        return _load_client().files


class AudioProxy(LazyProxy["Audio"]):
    @override
    def __load__(self) -> Audio:
        return _load_client().audio


class EvalsProxy(LazyProxy["Evals"]):
    @override
    def __load__(self) -> Evals:
        return _load_client().evals


class ImagesProxy(LazyProxy["Images"]):
    @override
    def __load__(self) -> Images:
        return _load_client().images


class ModelsProxy(LazyProxy["Models"]):
    @override
    def __load__(self) -> Models:
        return _load_client().models


class BatchesProxy(LazyProxy["Batches"]):
    @override
    def __load__(self) -> Batches:
        return _load_client().batches


class UploadsProxy(LazyProxy["Uploads"]):
    @override
    def __load__(self) -> Uploads:
        return _load_client().uploads


class WebhooksProxy(LazyProxy["Webhooks"]):
    @override
    def __load__(self) -> Webhooks:
        return _load_client().webhooks


class ResponsesProxy(LazyProxy["Responses"]):
    @override
    def __load__(self) -> Responses:
        return _load_client().responses


class EmbeddingsProxy(LazyProxy["Embeddings"]):
    @override
    def __load__(self) -> Embeddings:
        return _load_client().embeddings


class ContainersProxy(LazyProxy["Containers"]):
    @override
    def __load__(self) -> Containers:
        return _load_client().containers


class CompletionsProxy(LazyProxy["Completions"]):
    @override
    def __load__(self) -> Completions:
        return _load_client().completions


class ModerationsProxy(LazyProxy["Moderations"]):
    @override
    def __load__(self) -> Moderations:
        return _load_client().moderations


class FineTuningProxy(LazyProxy["FineTuning"]):
    @override
    def __load__(self) -> FineTuning:
        return _load_client().fine_tuning


class VectorStoresProxy(LazyProxy["VectorStores"]):
    @override
    def __load__(self) -> VectorStores:
        return _load_client().vector_stores


chat: Chat = ChatProxy().__as_proxied__()
beta: Beta = BetaProxy().__as_proxied__()
files: Files = FilesProxy().__as_proxied__()
audio: Audio = AudioProxy().__as_proxied__()
evals: Evals = EvalsProxy().__as_proxied__()
images: Images = ImagesProxy().__as_proxied__()
models: Models = ModelsProxy().__as_proxied__()
batches: Batches = BatchesProxy().__as_proxied__()
uploads: Uploads = UploadsProxy().__as_proxied__()
webhooks: Webhooks = WebhooksProxy().__as_proxied__()
responses: Responses = ResponsesProxy().__as_proxied__()
embeddings: Embeddings = EmbeddingsProxy().__as_proxied__()
containers: Containers = ContainersProxy().__as_proxied__()
completions: Completions = CompletionsProxy().__as_proxied__()
moderations: Moderations = ModerationsProxy().__as_proxied__()
fine_tuning: FineTuning = FineTuningProxy().__as_proxied__()
vector_stores: VectorStores = VectorStoresProxy().__as_proxied__()
