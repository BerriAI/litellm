#![allow(dead_code)]

use std::{
    collections::{HashMap, HashSet},
    net::SocketAddr,
    sync::{Arc, Mutex},
};

use qdrant_client::qdrant::{
    self, CollectionExists, CollectionExistsRequest, CollectionExistsResponse,
    CollectionOperationResponse, CreateCollection, CreateFieldIndexCollection, Filter, PointId,
    PointsOperationResponse, ScoredPoint, SearchPoints, SearchResponse, Value, Vector, Vectors,
    collections_server::{Collections, CollectionsServer},
    points_server::{Points, PointsServer},
};
use tokio::sync::oneshot;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{Request, Response, Status, transport::Server};

#[derive(Clone, Debug)]
pub struct StoredPoint {
    pub id: Option<PointId>,
    pub vector: Vec<f32>,
    pub payload: HashMap<String, Value>,
}

#[derive(Default)]
pub struct FakeState {
    pub collections: HashSet<String>,
    pub created_collections: Vec<CreateCollection>,
    pub field_indexes: Vec<CreateFieldIndexCollection>,
    pub points: Vec<StoredPoint>,
    pub upsert_waits: Vec<Option<bool>>,
    pub index_creations: usize,
    pub fail_field_index: bool,
}

#[derive(Clone)]
pub struct FakeQdrant {
    pub state: Arc<Mutex<FakeState>>,
    pub address: SocketAddr,
    shutdown: Arc<Mutex<Option<oneshot::Sender<()>>>>,
}

impl FakeQdrant {
    pub async fn start(state: FakeState) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let state = Arc::new(Mutex::new(state));
        let service = FakeService {
            state: state.clone(),
        };
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        tokio::spawn(async move {
            Server::builder()
                .add_service(CollectionsServer::new(service.clone()))
                .add_service(PointsServer::new(service))
                .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async {
                    let _ = shutdown_rx.await;
                })
                .await
                .unwrap();
        });
        Self {
            state,
            address,
            shutdown: Arc::new(Mutex::new(Some(shutdown_tx))),
        }
    }

    pub fn url(&self) -> String {
        format!("http://{}", self.address)
    }

    pub fn stop(&self) {
        self.shutdown
            .lock()
            .unwrap()
            .take()
            .unwrap()
            .send(())
            .unwrap();
    }

    pub fn insert_point(&self, point: StoredPoint) {
        self.state.lock().unwrap().points.push(point);
    }
}

#[derive(Clone)]
struct FakeService {
    state: Arc<Mutex<FakeState>>,
}

macro_rules! unimplemented_collections {
    ($($name:ident, $request:ty, $response:ty);* $(;)?) => {
        $(
            fn $name<'life0, 'async_trait>(
                &'life0 self,
                _: Request<$request>,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<
                            Output = Result<Response<$response>, Status>,
                        > + Send
                        + 'async_trait,
                >,
            >
            where
                'life0: 'async_trait,
                Self: 'async_trait,
            {
                Box::pin(async { Err(Status::unimplemented(stringify!($name))) })
            }
        )*
    };
}

macro_rules! unimplemented_points {
    ($($name:ident, $request:ty, $response:ty);* $(;)?) => {
        $(
            fn $name<'life0, 'async_trait>(
                &'life0 self,
                _: Request<$request>,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<
                            Output = Result<Response<$response>, Status>,
                        > + Send
                        + 'async_trait,
                >,
            >
            where
                'life0: 'async_trait,
                Self: 'async_trait,
            {
                Box::pin(async { Err(Status::unimplemented(stringify!($name))) })
            }
        )*
    };
}

#[tonic::async_trait]
impl Collections for FakeService {
    async fn create(
        &self,
        request: Request<CreateCollection>,
    ) -> Result<Response<CollectionOperationResponse>, Status> {
        let request = request.into_inner();
        let mut state = self.state.lock().unwrap();
        state.collections.insert(request.collection_name.clone());
        state.created_collections.push(request);
        Ok(Response::new(CollectionOperationResponse {
            result: true,
            ..Default::default()
        }))
    }

    async fn collection_exists(
        &self,
        request: Request<CollectionExistsRequest>,
    ) -> Result<Response<CollectionExistsResponse>, Status> {
        let exists = self
            .state
            .lock()
            .unwrap()
            .collections
            .contains(&request.into_inner().collection_name);
        Ok(Response::new(CollectionExistsResponse {
            result: Some(CollectionExists { exists }),
            ..Default::default()
        }))
    }

    unimplemented_collections!(
        get, qdrant::GetCollectionInfoRequest, qdrant::GetCollectionInfoResponse;
        list, qdrant::ListCollectionsRequest, qdrant::ListCollectionsResponse;
        update, qdrant::UpdateCollection, qdrant::CollectionOperationResponse;
        delete, qdrant::DeleteCollection, qdrant::CollectionOperationResponse;
        update_aliases, qdrant::ChangeAliases, qdrant::CollectionOperationResponse;
        list_collection_aliases, qdrant::ListCollectionAliasesRequest, qdrant::ListAliasesResponse;
        list_aliases, qdrant::ListAliasesRequest, qdrant::ListAliasesResponse;
        collection_cluster_info, qdrant::CollectionClusterInfoRequest, qdrant::CollectionClusterInfoResponse;
        update_collection_cluster_setup, qdrant::UpdateCollectionClusterSetupRequest, qdrant::UpdateCollectionClusterSetupResponse;
        create_shard_key, qdrant::CreateShardKeyRequest, qdrant::CreateShardKeyResponse;
        delete_shard_key, qdrant::DeleteShardKeyRequest, qdrant::DeleteShardKeyResponse;
        list_shard_keys, qdrant::ListShardKeysRequest, qdrant::ListShardKeysResponse;
    );
}

#[tonic::async_trait]
impl Points for FakeService {
    async fn create_field_index(
        &self,
        request: Request<CreateFieldIndexCollection>,
    ) -> Result<Response<PointsOperationResponse>, Status> {
        let mut state = self.state.lock().unwrap();
        state.index_creations += 1;
        state.field_indexes.push(request.into_inner());
        if state.fail_field_index {
            return Err(Status::internal("field index failure"));
        }
        Ok(Response::new(PointsOperationResponse::default()))
    }

    async fn upsert(
        &self,
        request: Request<qdrant::UpsertPoints>,
    ) -> Result<Response<PointsOperationResponse>, Status> {
        let request = request.into_inner();
        let mut state = self.state.lock().unwrap();
        state.upsert_waits.push(request.wait);
        for point in request.points {
            let stored = StoredPoint {
                id: point.id.clone(),
                vector: dense_vector(point.vectors)?,
                payload: point.payload,
            };
            if let Some(existing) = state
                .points
                .iter_mut()
                .find(|existing| existing.id == stored.id)
            {
                *existing = stored;
            } else {
                state.points.push(stored);
            }
        }
        Ok(Response::new(PointsOperationResponse::default()))
    }

    async fn search(
        &self,
        request: Request<SearchPoints>,
    ) -> Result<Response<SearchResponse>, Status> {
        let request = request.into_inner();
        let key_filter = keyword_filter(request.filter.as_ref());
        let state = self.state.lock().unwrap();
        let mut results = state
            .points
            .iter()
            .filter(|point| {
                key_filter.as_ref().is_none_or(|(field, expected)| {
                    point
                        .payload
                        .get(field)
                        .and_then(|value| {
                            let value: serde_json::Value = value.clone().into();
                            value
                                .as_str()
                                .map(str::to_owned)
                                .or_else(|| value.as_i64().map(|value| value.to_string()))
                        })
                        .is_some_and(|value| value == *expected)
                })
            })
            .map(|point| ScoredPoint {
                id: point.id.clone(),
                payload: point.payload.clone(),
                score: cosine(&request.vector, &point.vector),
                ..Default::default()
            })
            .collect::<Vec<_>>();
        results.sort_by(|left, right| right.score.total_cmp(&left.score));
        results.truncate(request.limit as usize);
        Ok(Response::new(SearchResponse {
            result: results,
            ..Default::default()
        }))
    }

    unimplemented_points!(
        delete, qdrant::DeletePoints, qdrant::PointsOperationResponse;
        get, qdrant::GetPoints, qdrant::GetResponse;
        update_vectors, qdrant::UpdatePointVectors, qdrant::PointsOperationResponse;
        delete_vectors, qdrant::DeletePointVectors, qdrant::PointsOperationResponse;
        set_payload, qdrant::SetPayloadPoints, qdrant::PointsOperationResponse;
        overwrite_payload, qdrant::SetPayloadPoints, qdrant::PointsOperationResponse;
        delete_payload, qdrant::DeletePayloadPoints, qdrant::PointsOperationResponse;
        clear_payload, qdrant::ClearPayloadPoints, qdrant::PointsOperationResponse;
        delete_field_index, qdrant::DeleteFieldIndexCollection, qdrant::PointsOperationResponse;
        create_vector_name, qdrant::CreateVectorNameRequest, qdrant::PointsOperationResponse;
        delete_vector_name, qdrant::DeleteVectorNameRequest, qdrant::PointsOperationResponse;
        search_batch, qdrant::SearchBatchPoints, qdrant::SearchBatchResponse;
        search_groups, qdrant::SearchPointGroups, qdrant::SearchGroupsResponse;
        scroll, qdrant::ScrollPoints, qdrant::ScrollResponse;
        recommend, qdrant::RecommendPoints, qdrant::RecommendResponse;
        recommend_batch, qdrant::RecommendBatchPoints, qdrant::RecommendBatchResponse;
        recommend_groups, qdrant::RecommendPointGroups, qdrant::RecommendGroupsResponse;
        discover, qdrant::DiscoverPoints, qdrant::DiscoverResponse;
        discover_batch, qdrant::DiscoverBatchPoints, qdrant::DiscoverBatchResponse;
        count, qdrant::CountPoints, qdrant::CountResponse;
        update_batch, qdrant::UpdateBatchPoints, qdrant::UpdateBatchResponse;
        query, qdrant::QueryPoints, qdrant::QueryResponse;
        query_batch, qdrant::QueryBatchPoints, qdrant::QueryBatchResponse;
        query_groups, qdrant::QueryPointGroups, qdrant::QueryGroupsResponse;
        facet, qdrant::FacetCounts, qdrant::FacetResponse;
        search_matrix_pairs, qdrant::SearchMatrixPoints, qdrant::SearchMatrixPairsResponse;
        search_matrix_offsets, qdrant::SearchMatrixPoints, qdrant::SearchMatrixOffsetsResponse;
    );
}

fn dense_vector(vectors: Option<Vectors>) -> Result<Vec<f32>, Status> {
    let Some(Vectors {
        vectors_options:
            Some(qdrant::vectors::VectorsOptions::Vector(Vector {
                vector: Some(qdrant::vector::Vector::Dense(qdrant::DenseVector { data })),
                ..
            })),
    }) = vectors
    else {
        return Err(Status::invalid_argument("expected dense vector"));
    };
    Ok(data)
}

fn keyword_filter(filter: Option<&Filter>) -> Option<(String, String)> {
    filter?
        .must
        .iter()
        .find_map(|condition| match condition.condition_one_of.as_ref()? {
            qdrant::condition::ConditionOneOf::Field(field) => {
                let qdrant::r#match::MatchValue::Keyword(value) =
                    field.r#match.as_ref()?.match_value.as_ref()?
                else {
                    return None;
                };
                Some((field.key.clone(), value.clone()))
            }
            _ => None,
        })
}

fn cosine(left: &[f32], right: &[f32]) -> f32 {
    let dot = left
        .iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum::<f32>();
    let left_norm = left.iter().map(|value| value * value).sum::<f32>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f32>().sqrt();
    dot / (left_norm * right_norm)
}
