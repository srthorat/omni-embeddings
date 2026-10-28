use serde::{Deserialize, Serialize};

/// Supported Model Architectures in Omni-Embeddings
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelArchitecture {
    /// Google EmbeddingGemma 2 (740M Multimodal Transformer, 24 layers, 768-dim)
    EmbeddingGemma2,
    /// Google EmbeddingGemma 300M (308M Gemma 3 Architecture, 18 layers, dual RoPE)
    EmbeddingGemma300M,
    /// NomicBERT v1.5 (137M parameters, 8192 context, Matryoshka MRL 64..768)
    NomicBert,
    /// BAAI BGE-Visualized-M3 (560M parameters, 1024-dim, composed Image+Text retrieval)
    BgeVisualizedM3,
}

/// Modalities supported by the model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SupportedModality {
    TextOnly,
    TextAndImage,
    TextVisionAudio,
}

/// Pooling Strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PoolingStrategy {
    Mean,
    Cls,
    LastToken,
}

/// Complete Model Specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    pub id: String,
    pub name: String,
    pub architecture: ModelArchitecture,
    pub parameter_count_millions: usize,
    pub default_dimension: usize,
    pub supported_dimensions: Vec<usize>,
    pub max_context_length: usize,
    pub modality: SupportedModality,
    pub pooling: PoolingStrategy,
}

impl ModelSpec {
    /// Google EmbeddingGemma-2 (740M)
    pub fn gemma_2_740m() -> Self {
        Self {
            id: "google/embeddinggemma-2".to_string(),
            name: "EmbeddingGemma 2 (740M Multimodal)".to_string(),
            architecture: ModelArchitecture::EmbeddingGemma2,
            parameter_count_millions: 740,
            default_dimension: 768,
            supported_dimensions: vec![768],
            max_context_length: 262_144,
            modality: SupportedModality::TextVisionAudio,
            pooling: PoolingStrategy::Mean,
        }
    }

    /// Google EmbeddingGemma-300M (308M Gemma 3 Architecture)
    pub fn gemma_3_300m() -> Self {
        Self {
            id: "google/embeddinggemma-300m".to_string(),
            name: "EmbeddingGemma 300M (Gemma 3 Architecture)".to_string(),
            architecture: ModelArchitecture::EmbeddingGemma300M,
            parameter_count_millions: 308,
            default_dimension: 768,
            supported_dimensions: vec![512, 768],
            max_context_length: 32_768,
            modality: SupportedModality::TextOnly,
            pooling: PoolingStrategy::Mean,
        }
    }

    /// Nomic Embed Text v1.5 (137M with Matryoshka Representation Learning)
    pub fn nomic_embed_v1_5() -> Self {
        Self {
            id: "nomic-ai/nomic-embed-text-v1.5".to_string(),
            name: "Nomic Embed Text v1.5 (137M MRL)".to_string(),
            architecture: ModelArchitecture::NomicBert,
            parameter_count_millions: 137,
            default_dimension: 768,
            supported_dimensions: vec![64, 128, 256, 512, 768],
            max_context_length: 8_192,
            modality: SupportedModality::TextOnly,
            pooling: PoolingStrategy::Mean,
        }
    }

    /// BAAI BGE-Visualized-M3 (560M Multi-Modal Composed Retrieval)
    pub fn bge_visualized_m3() -> Self {
        Self {
            id: "BAAI/bge-visualized-m3".to_string(),
            name: "BGE Visualized M3 (560M Composed Image-Text)".to_string(),
            architecture: ModelArchitecture::BgeVisualizedM3,
            parameter_count_millions: 560,
            default_dimension: 1024,
            supported_dimensions: vec![1024],
            max_context_length: 8_192,
            modality: SupportedModality::TextAndImage,
            pooling: PoolingStrategy::Cls,
        }
    }

    /// Return all supported models in registry
    pub fn all_models() -> Vec<Self> {
        vec![
            Self::gemma_2_740m(),
            Self::gemma_3_300m(),
            Self::nomic_embed_v1_5(),
            Self::bge_visualized_m3(),
        ]
    }

    /// Resolve model specification from identifier or alias
    pub fn from_identifier(identifier: &str) -> Self {
        let lower = identifier.to_lowercase();
        if lower.contains("300m") || lower.contains("gemma-300m") {
            Self::gemma_3_300m()
        } else if lower.contains("nomic") {
            Self::nomic_embed_v1_5()
        } else if lower.contains("bge") || lower.contains("visualized") {
            Self::bge_visualized_m3()
        } else {
            Self::gemma_2_740m()
        }
    }

    /// Apply Matryoshka Representation Learning (MRL) dimension truncation & re-normalization
    pub fn apply_matryoshka_dimension(&self, mut vector: Vec<f32>, target_dim: Option<usize>) -> Vec<f32> {
        let dim = match target_dim {
            Some(d) if self.supported_dimensions.contains(&d) && d <= vector.len() => d,
            _ => self.default_dimension.min(vector.len()),
        };

        vector.truncate(dim);

        // L2 Re-normalization
        let norm: f32 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut vector {
                *v /= norm;
            }
        }
        vector
    }
}

impl Default for ModelSpec {
    fn default() -> Self {
        Self::gemma_2_740m()
    }
}
