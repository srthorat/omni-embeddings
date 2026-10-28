use crate::models::ModelSpec;
use crate::TEIStandardEmbeddingEngine;
use axum::{
    extract::State,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<TEIStandardEmbeddingEngine>,
}

pub fn create_router(engine: Arc<TEIStandardEmbeddingEngine>) -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/v1/embeddings", post(create_embeddings))
        .route("/embed", post(tei_embed))
        .route("/models", get(list_models))
        .route("/v1/models", get(list_models))
        .with_state(AppState { engine })
}

async fn health_check(State(state): State<AppState>) -> impl IntoResponse {
    let (hits, misses, entries) = state.engine.batcher.cache.stats();
    let hit_rate = state.engine.batcher.cache.hit_rate();
    let model = &state.engine.batcher.model_spec;

    Json(serde_json::json!({
        "status": "healthy",
        "runtime": "native-rust-omni-embeddings",
        "model_id": model.id,
        "model_name": model.name,
        "dimension": model.default_dimension,
        "max_context": model.max_context_length,
        "cache": {
            "type": "W-TinyLFU (Moka)",
            "entries": entries,
            "hits": hits,
            "misses": misses,
            "hit_rate": format!("{:.1}%", hit_rate)
        }
    }))
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum InputText {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Deserialize)]
pub struct OpenAIEmbeddingRequest {
    pub input: InputText,
    pub model: Option<String>,
    pub dimensions: Option<usize>,
}

#[derive(Serialize)]
pub struct OpenAIEmbeddingItem {
    pub object: String,
    pub index: usize,
    pub embedding: Vec<f32>,
}

#[derive(Serialize)]
pub struct OpenAIEmbeddingResponse {
    pub object: String,
    pub data: Vec<OpenAIEmbeddingItem>,
    pub model: String,
    pub usage: serde_json::Value,
}

async fn create_embeddings(
    State(state): State<AppState>,
    Json(payload): Json<OpenAIEmbeddingRequest>,
) -> impl IntoResponse {
    let texts = match payload.input {
        InputText::Single(t) => vec![t],
        InputText::Multiple(list) => list,
    };

    let total_tokens: usize = texts.iter().map(|t| t.split_whitespace().count()).sum();
    let vectors = state
        .engine
        .embed_batch_with_dimensions(&texts, payload.dimensions)
        .await
        .unwrap_or_else(|_| vec![vec![0.0f32; state.engine.batcher.model_spec.default_dimension]; texts.len()]);

    let data: Vec<OpenAIEmbeddingItem> = vectors
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| OpenAIEmbeddingItem {
            object: "embedding".to_string(),
            index,
            embedding,
        })
        .collect();

    Json(OpenAIEmbeddingResponse {
        object: "list".to_string(),
        data,
        model: payload.model.unwrap_or_else(|| state.engine.batcher.model_spec.id.clone()),
        usage: serde_json::json!({
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens
        }),
    })
}

#[derive(Deserialize)]
pub struct TEIEmbedRequest {
    pub inputs: serde_json::Value,
    #[serde(default)]
    pub normalize: bool,
    #[serde(default)]
    pub truncate: bool,
}

async fn tei_embed(
    State(state): State<AppState>,
    Json(payload): Json<TEIEmbedRequest>,
) -> impl IntoResponse {
    let texts: Vec<String> = if let Some(s) = payload.inputs.as_str() {
        vec![s.to_string()]
    } else if let Some(arr) = payload.inputs.as_array() {
        arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect()
    } else {
        vec![]
    };

    let vectors = state
        .engine
        .embed_batch(&texts)
        .await
        .unwrap_or_else(|_| vec![vec![0.0f32; state.engine.batcher.model_spec.default_dimension]; texts.len()]);

    Json(vectors)
}

async fn list_models() -> impl IntoResponse {
    let models = ModelSpec::all_models();
    let data: Vec<serde_json::Value> = models
        .into_iter()
        .map(|m| {
            serde_json::json!({
                "id": m.id,
                "object": "model",
                "name": m.name,
                "dimension": m.default_dimension,
                "supported_dimensions": m.supported_dimensions,
                "max_context": m.max_context_length,
                "parameters_m": m.parameter_count_millions
            })
        })
        .collect();

    Json(serde_json::json!({
        "object": "list",
        "data": data
    }))
}
