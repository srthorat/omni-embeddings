use omni_embeddings::models::ModelSpec;
use omni_embeddings::{RaggedBatch, TEIStandardEmbeddingEngine};

#[tokio::test]
async fn test_ragged_batch_packing() {
    let s1 = vec![1, 2, 3, 4];
    let s2 = vec![5, 6];
    let s3 = vec![7, 8, 9];

    let ragged = RaggedBatch::from_token_sequences(&[s1, s2, s3]);
    assert_eq!(ragged.batch_size, 3);
    assert_eq!(ragged.total_tokens(), 9); // ZERO padding tokens!
    assert_eq!(ragged.cu_seqlens, vec![0, 4, 6, 9]);
    assert_eq!(ragged.max_seqlen, 4);
}

#[tokio::test]
async fn test_moka_cache_and_batcher() {
    let engine = TEIStandardEmbeddingEngine::new(4096, 64, ModelSpec::gemma_2_740m());

    let texts = vec![
        "Self-hosted high throughput embedding inference".to_string(),
        "Production grade W-TinyLFU cache".to_string(),
    ];

    // Cold pass
    let r1 = engine.embed_batch(&texts).await.unwrap();
    assert_eq!(r1.len(), 2);
    assert_eq!(r1[0].len(), 768);

    // Warm pass - Instant Moka hit
    let (hits_before, _, _) = engine.batcher.cache.stats();
    let r2 = engine.embed_text(&texts[0]).await.unwrap();
    assert_eq!(r2.len(), 768);
    let (hits_after, _, _) = engine.batcher.cache.stats();
    assert_eq!(hits_after, hits_before + 1);
}

#[tokio::test]
async fn test_matryoshka_dimension_scaling() {
    let nomic_spec = ModelSpec::nomic_embed_v1_5();
    assert_eq!(nomic_spec.default_dimension, 768);
    assert!(nomic_spec.supported_dimensions.contains(&256));

    let full_vec = vec![0.5f32; 768];
    let sliced_256 = nomic_spec.apply_matryoshka_dimension(full_vec.clone(), Some(256));
    assert_eq!(sliced_256.len(), 256);

    let norm: f32 = sliced_256.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-4); // Verified re-normalized unit vector
}

#[tokio::test]
async fn test_model_registry_routing() {
    let gemma2 = ModelSpec::from_identifier("google/embeddinggemma-2");
    assert_eq!(gemma2.default_dimension, 768);
    assert_eq!(gemma2.parameter_count_millions, 740);

    let gemma3 = ModelSpec::from_identifier("google/embeddinggemma-300m");
    assert_eq!(gemma3.parameter_count_millions, 308);

    let nomic = ModelSpec::from_identifier("nomic-ai/nomic-embed-text-v1.5");
    assert_eq!(nomic.parameter_count_millions, 137);

    let bge = ModelSpec::from_identifier("BAAI/bge-visualized-m3");
    assert_eq!(bge.default_dimension, 1024);
}
