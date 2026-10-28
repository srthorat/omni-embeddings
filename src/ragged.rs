/// Unpadded Ragged Batch (TEI Gold Standard)
/// Sequences are flattened into a continuous 1D token array with cumulative sequence length offsets (cu_seqlens),
/// completely eliminating zero-padding compute waste.
#[derive(Debug, Clone)]
pub struct RaggedBatch {
    pub flattened_tokens: Vec<u32>,
    pub cu_seqlens: Vec<usize>,
    pub batch_size: usize,
    pub max_seqlen: usize,
}

impl RaggedBatch {
    /// Pack variable-length token sequences into a contiguous unpadded 1D ragged tensor
    pub fn from_token_sequences(sequences: &[Vec<u32>]) -> Self {
        let batch_size = sequences.len();
        let total_tokens: usize = sequences.iter().map(|s| s.len()).sum();
        let mut flattened_tokens = Vec::with_capacity(total_tokens);
        let mut cu_seqlens = Vec::with_capacity(batch_size + 1);
        cu_seqlens.push(0);

        let mut max_seqlen = 0;
        let mut running_offset = 0;

        for seq in sequences {
            max_seqlen = max_seqlen.max(seq.len());
            running_offset += seq.len();
            cu_seqlens.push(running_offset);
            flattened_tokens.extend_from_slice(seq);
        }

        Self {
            flattened_tokens,
            cu_seqlens,
            batch_size,
            max_seqlen,
        }
    }

    /// Total active tokens across all sequences (Zero padded tokens = 0)
    pub fn total_tokens(&self) -> usize {
        self.flattened_tokens.len()
    }
}
