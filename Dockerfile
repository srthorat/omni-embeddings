# Multi-stage build for minimal distroless image
FROM rust:1.80-bullseye as builder

WORKDIR /usr/src/omni-embeddings
COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN cargo build --release

FROM gcr.io/distroless/cc-debian11

COPY --from=builder /usr/src/omni-embeddings/target/release/omni-embeddings /usr/local/bin/omni-embeddings

ENV PORT=8000
EXPOSE 8000

ENTRYPOINT ["omni-embeddings"]
CMD ["--port", "8000", "--host", "0.0.0.0"]
