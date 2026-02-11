use crate::{Result, VoxCpmError};
use std::collections::HashSet;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct VoxCpmTokenizer {
    inner: tokenizers::Tokenizer,
    // Python reference wraps the tokenizer to split multi-character CJK vocab tokens
    // (e.g. "你好") into per-character tokens ("你", "好").
    multichar_chinese_tokens: HashSet<String>,
    unk_id: Option<u32>,
}

impl VoxCpmTokenizer {
    pub fn from_tokenizer_json(path: &Path) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path).map_err(VoxCpmError::Tokenizer)?;

        let vocab = inner.get_vocab(true);
        let multichar_chinese_tokens = vocab
            .keys()
            .filter(|tok| is_multichar_chinese_token(tok))
            .cloned()
            .collect::<HashSet<_>>();

        // Best-effort: different tokenizers use different unk token spellings.
        let unk_id = ["<unk>", "[UNK]", "<|unk|>", "<UNK>"]
            .into_iter()
            .find_map(|t| vocab.get(t).copied());

        Ok(Self {
            inner,
            multichar_chinese_tokens,
            unk_id,
        })
    }

    pub fn encode_ids(&self, text: &str) -> Result<Vec<u32>> {
        // Match the Python reference behavior:
        // - no automatic special tokens
        // - split multi-character Chinese tokens emitted by the tokenizer
        let enc = self
            .inner
            .encode(text, false)
            .map_err(VoxCpmError::Tokenizer)?;

        let ids = enc.get_ids();
        let toks = enc.get_tokens();
        debug_assert_eq!(ids.len(), toks.len());

        let mut out = Vec::with_capacity(ids.len());
        for (id, tok) in ids.iter().copied().zip(toks.iter()) {
            // Python strips the SentencePiece-like word boundary marker before checking.
            let clean = tok.replace('▁', "");
            if self.multichar_chinese_tokens.contains(&clean) {
                for ch in clean.chars() {
                    let s = ch.to_string();
                    match self.inner.token_to_id(&s) {
                        Some(cid) => out.push(cid as u32),
                        None => {
                            let unk = self.unk_id.ok_or_else(|| {
                                VoxCpmError::InvalidArg(
                                    "tokenizer is missing an unk token id; cannot map split Chinese characters"
                                        .into(),
                                )
                            })?;
                            out.push(unk);
                        }
                    }
                }
            } else {
                out.push(id as u32)
            }
        }
        Ok(out)
    }
}

fn is_multichar_chinese_token(tok: &str) -> bool {
    let mut chars = tok.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !is_cjk_unified_ideograph(first) {
        return false;
    }
    let mut count = 1usize;
    for ch in chars {
        if !is_cjk_unified_ideograph(ch) {
            return false;
        }
        count += 1;
    }
    count >= 2
}

fn is_cjk_unified_ideograph(c: char) -> bool {
    ('\u{4E00}'..='\u{9FFF}').contains(&c)
}
