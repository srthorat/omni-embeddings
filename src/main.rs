use clap::Parser;
use omni_embeddings::models::ModelSpec;
use omni_embeddings::routes::create_router;
use omni_embeddings::TEIStandardEmbeddingEngine;
use std::net::SocketAddr;
use std::sync::Arc;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser, Debug)]
#[command(
    name = "omni-embeddings",
    author = "Streaming AI Team",
    version = "0.2.0",
    about = "High-Throughput Native Rust Embedding Server with Continuous Ragged Dynamic Batching and W-TinyLFU Moka Cache"
)]
struct Args {
    /// Port to listen on
    #[arg(short, long, env = "PORT", default_value_t = 8000)]
    port: u16,

    /// Host IP address to bind
    #[arg(long, env = "HOST", default_value = "0.0.0.0")]
    host: String,

    /// Target model identifier (google/embeddinggemma-2, google/embeddinggemma-300m, nomic-ai/nomic-embed-text-v1.5, BAAI/bge-visualized-m3)
    #[arg(short, long, env = "MODEL_ID", default_value = "google/embeddinggemma-2")]
    model_id: String,

    /// Maximum total tokens per dynamic continuous batch
    #[arg(long, env = "MAX_BATCH_TOKENS", default_value_t = 16384)]
    max_batch_tokens: usize,

    /// Maximum concurrent requests per batch
    #[arg(long, env = "MAX_BATCH_SIZE", default_value_t = 128)]
    max_batch_size: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "omni_embeddings=info,tower_http=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let args = Args::parse();
    let model_spec = ModelSpec::from_identifier(&args.model_id);

    tracing::info!("═════════════════════════════════════════════════════════════════");
    tracing::info!("🚀 Starting Omni-Embeddings High-Throughput Server (v0.2.0)");
    tracing::info!("   Model ID:         {}", model_spec.id);
    tracing::info!("   Architecture:     {:?} ({}M params)", model_spec.architecture, model_spec.parameter_count_millions);
    tracing::info!("   Dimension:        {} (Supported: {:?})", model_spec.default_dimension, model_spec.supported_dimensions);
    tracing::info!("   Max Context:      {} tokens", model_spec.max_context_length);
    tracing::info!("   Cache:            W-TinyLFU Moka In-Memory (100,000 vectors bounded)");
    tracing::info!("   Dynamic Batching: Continuous Ragged (max {} tokens)", args.max_batch_tokens);
    tracing::info!("═════════════════════════════════════════════════════════════════");

    let engine = Arc::new(TEIStandardEmbeddingEngine::new(
        args.max_batch_tokens,
        args.max_batch_size,
        model_spec,
    ));

    let app = create_router(engine);
    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    tracing::info!("⚡ Server listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
