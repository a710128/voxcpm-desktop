use std::path::PathBuf;

use voxcpm::VoxCpmTokenizer;

#[test]
#[ignore]
fn tokenizer_masks_multichar_chinese_tokens_like_python() {
    let Ok(model_dir) = std::env::var("VOXCPM_MODEL_DIR") else {
        eprintln!(
            "VOXCPM_MODEL_DIR is not set; set it to a model dir containing tokenizer.json to run this test"
        );
        return;
    };
    let tokenizer_json = PathBuf::from(model_dir).join("tokenizer.json");

    if !tokenizer_json.is_file() {
        panic!(
            "tokenizer.json is missing at {:?} (VOXCPM_MODEL_DIR should point to a model dir containing tokenizer.json)",
            tokenizer_json
        );
    }

    let tok = VoxCpmTokenizer::from_tokenizer_json(&tokenizer_json).unwrap_or_else(|e| {
        panic!(
            "failed to load VoxCpmTokenizer from tokenizer.json at {:?}: {e}",
            tokenizer_json
        )
    });

    // Ensure we match the reference choice of *not* adding special tokens.
    let raw = tokenizers::Tokenizer::from_file(&tokenizer_json).unwrap_or_else(|e| {
        panic!(
            "failed to load reference tokenizers::Tokenizer from file {:?}: {e}",
            tokenizer_json
        )
    });
    let baseline = raw.encode("hello", false).unwrap();
    assert_eq!(
        tok.encode_ids("hello").unwrap(),
        baseline
            .get_ids()
            .iter()
            .map(|&id| id as u32)
            .collect::<Vec<_>>()
    );

    // Try to find a vocab entry like "你好" which tokenizes as a single token,
    // and whose individual chars exist as tokens.
    let vocab = raw.get_vocab(true);
    let mut found = None;
    for t in vocab.keys() {
        if !is_multichar_chinese_token(t) {
            continue;
        }
        let enc = raw.encode(t.as_str(), false).unwrap();
        if enc.get_tokens().len() != 1 {
            continue;
        }
        let emitted = enc.get_tokens()[0].replace('▁', "");
        if emitted != *t {
            continue;
        }

        let mut expected = Vec::new();
        let mut ok = true;
        for ch in t.chars() {
            let s = ch.to_string();
            let Some(id) = raw.token_to_id(&s) else {
                ok = false;
                break;
            };
            expected.push(id as u32);
        }
        if !ok {
            continue;
        }
        found = Some((t.clone(), expected));
        break;
    }

    let Some((case, expected)) = found else {
        eprintln!("no suitable multi-char Chinese token found; tokenizer may already split these or lacks single-char entries");
        return;
    };

    assert_eq!(tok.encode_ids(&case).unwrap(), expected);
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
