pub mod batcher;
pub mod cache;
pub mod models;
pub mod ragged;
pub mod routes;

use std::sync::Arc;
pub use batcher::DynamicBatcher;
pub use cache::MokaVectorCache;
pub use models::{ModelArchitecture, ModelSpec, PoolingStrategy, SupportedModality};
pub use ragged::RaggedBatch;

/// High-level TEI-standard Text Embedding Service
#[derive(Clone)]
pub struct TEIStandardEmbeddingEngine {
    pub batcher: Arc<DynamicBatcher>,
}

impl TEIStandardEmbeddingEngine {
    pub fn new(max_batch_tokens: usize, max_batch_size: usize, model_spec: ModelSpec) -> Self {
        Self {
            batcher: Arc::new(DynamicBatcher::new(max_batch_tokens, max_batch_size, model_spec)),
        }
    }

    /// Embed an individual text sequence
    pub async fn embed_text(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        self.embed_text_with_dimensions(text, None).await
    }

    /// Embed an individual text sequence with optional Matryoshka dimension
    pub async fn embed_text_with_dimensions(&self, text: &str, dimensions: Option<usize>) -> anyhow::Result<Vec<f32>> {
        // Fast tokenization mapping bytes/words to token IDs
        let tokens: Vec<u32> = text
            .bytes()
            .map(|b| b as u32)
            .take(512)
            .collect();

        self.batcher.submit(text.to_string(), tokens, dimensions).await
    }

    /// Embed a batch of texts concurrently via TEI continuous dynamic batcher
    pub async fn embed_batch(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        self.embed_batch_with_dimensions(texts, None).await
    }

    /// Embed a batch of texts with Matryoshka dimension truncation
    pub async fn embed_batch_with_dimensions(
        &self,
        texts: &[String],
        dimensions: Option<usize>,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        let futures: Vec<_> = texts
            .iter()
            .map(|text| self.embed_text_with_dimensions(text, dimensions))
            .collect();

        futures::future::try_join_all(futures).await
    }
}

impl Default for TEIStandardEmbeddingEngine {
    fn default() -> Self {
        Self::new(16384, 128, ModelSpec::default())
    }
}
