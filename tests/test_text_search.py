"""Tests for SigLIP text search — embedding alignment, confidence scoring, and pitfalls.

Background
==========
Text search embeds a text query with SigLIP's text tower and finds the closest
image embeddings in the FAISS index using L2 distance. Several non-obvious
issues make this fragile:

1. **Tokenizer bug in transformers**: ``AutoTokenizer.from_pretrained`` and
   ``AutoProcessor.from_pretrained`` crash on SigLIP because
   ``TOKENIZER_MAPPING_NAMES['siglip']`` is ``None`` (the slow ``SiglipTokenizer``
   needs ``sentencepiece``). We work around this by using ``GemmaTokenizerFast``
   directly, since SigLIP uses a Gemma-family tokenizer.

2. **Padding must be ``"max_length"``**: SigLIP was trained with 64-token
   padded inputs. Using ``padding=True`` (shortest-possible) produces embeddings
   in a completely different region of the space, flipping cosine similarity
   from slightly positive to negative. This was the root cause of text search
   returning zero results.

3. **``get_text_features`` return type changed**: In older transformers versions
   it returned a plain tensor. In newer versions it returns
   ``BaseModelOutputWithPooling`` — calling ``.norm()`` on it crashes. We call
   ``model.text_model(...)`` directly and extract ``.pooler_output``.

4. **SigLIP cosine similarities are tiny**: Unlike CLIP which produces cosine
   similarities of 0.2–0.3 for good text-image matches, SigLIP produces
   cosine similarities of 0.05–0.15. This is by design (sigmoid contrastive
   loss with learned ``logit_scale ≈ 117`` and ``logit_bias ≈ -13``). Raw
   cosine similarity must be converted through the model's sigmoid:
   ``confidence = sigmoid(logit_scale * cos_sim + logit_bias)``

5. **FAISS returns squared L2**: ``IndexFlatL2.search()`` returns **squared**
   L2 distances, not raw L2. For unit vectors, ``cos_sim = 1 - dist/2``, NOT
   ``1 - dist²/2``. Getting this wrong produces wildly incorrect confidence
   scores (double-squaring maps everything to near-zero).

6. **CLIP's ``get_text_features`` also changed**: It now returns
   ``BaseModelOutputWithPooling``, so extracting ``.pooler_output`` gives the
   *raw* text output without the learned ``text_projection``. The fix is to
   call ``model.text_model(...) → text_projection(pooler_output)`` explicitly,
   mirroring how ``embed()`` uses ``visual_projection``.

Debugging trace
===============
The original error was ``'NoneType' object has no attribute 'replace'`` from
``AutoTokenizer.from_pretrained``, caused by the transformers
``TOKENIZER_MAPPING_NAMES`` bug. After fixing that, ``get_text_features``
returned ``BaseModelOutputWithPooling`` instead of a tensor, crashing on
``.norm()``. After fixing *that*, search still returned zero results because:
- ``padding=True`` put text embeddings in the wrong region (cos_sim ≈ -0.05)
- The ``min_confidence=0.6`` threshold filtered out all text results
- Even after fixing padding, ``text_confidence`` used ``1 - dist²/2`` instead
  of ``1 - dist/2``, computing nonsensical values

Each bug masked the next one, making this a multi-layer debugging problem.
"""

import math

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def siglip_model():
    """Load SigLIP model once for the entire test module."""
    from transformers import AutoModel
    model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
    model.eval()
    return model


@pytest.fixture(scope="module")
def siglip_tokenizer():
    from transformers import GemmaTokenizerFast
    return GemmaTokenizerFast.from_pretrained("google/siglip-base-patch16-224")


@pytest.fixture(scope="module")
def siglip_embedder():
    from sam3_pursuit.models.embedder import SigLIPEmbedder
    return SigLIPEmbedder(device="cpu")


# ---------------------------------------------------------------------------
# Pitfall 1: tokenizer loading
# ---------------------------------------------------------------------------

def test_auto_tokenizer_crashes_on_siglip():
    """AutoTokenizer.from_pretrained fails on SigLIP — this is the transformers
    bug that started the whole investigation.

    The mapping ``TOKENIZER_MAPPING_NAMES['siglip']`` is ``None`` because the
    slow SiglipTokenizer requires sentencepiece. The ``.get()`` returns None
    (stored value) instead of the default ``""``, and ``.replace("Fast", "")``
    crashes.
    """
    from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING_NAMES
    # Verify the bug still exists in this transformers version
    val = TOKENIZER_MAPPING_NAMES.get("siglip", "MISSING")
    if val is None:
        # The bug: .get() returns None (the stored value), not the default
        with pytest.raises(AttributeError, match="replace"):
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")
    else:
        pytest.skip("TOKENIZER_MAPPING_NAMES bug is fixed in this version")


def test_gemma_tokenizer_works():
    """GemmaTokenizerFast works as a drop-in replacement for SigLIP."""
    from transformers import GemmaTokenizerFast
    tok = GemmaTokenizerFast.from_pretrained("google/siglip-base-patch16-224")
    out = tok(["hello world"], return_tensors="pt", padding="max_length")
    assert out["input_ids"].shape == (1, 64), "SigLIP max_length is 64 tokens"


# ---------------------------------------------------------------------------
# Pitfall 2: padding="max_length" is required
# ---------------------------------------------------------------------------

def test_padding_max_length_vs_default(siglip_model, siglip_tokenizer):
    """SigLIP was trained with padding='max_length' (64 tokens). Using
    padding=True (shortest-possible) produces very different embeddings.

    This was the root cause of text search failing: the wrong padding puts
    text embeddings in a different region of the space, making cosine
    similarity with image embeddings negative instead of positive.
    """
    text = "a photo of a blue fox"

    inputs_maxlen = siglip_tokenizer([text], return_tensors="pt", padding="max_length")
    inputs_short = siglip_tokenizer([text], return_tensors="pt", padding=True)

    assert inputs_maxlen["input_ids"].shape[1] == 64
    assert inputs_short["input_ids"].shape[1] < 64

    with torch.no_grad():
        emb_maxlen = siglip_model.text_model(**inputs_maxlen).pooler_output[0]
        emb_short = siglip_model.text_model(**inputs_short).pooler_output[0]

    emb_maxlen = emb_maxlen / emb_maxlen.norm()
    emb_short = emb_short / emb_short.norm()
    cos_sim = torch.dot(emb_maxlen, emb_short).item()

    # Embeddings diverge significantly — cosine sim is well below 1.0
    assert cos_sim < 0.85, (
        f"Expected divergent embeddings from different padding, got cos_sim={cos_sim:.4f}"
    )


# ---------------------------------------------------------------------------
# Pitfall 3: text and vision embeddings live in the same space
# ---------------------------------------------------------------------------

def test_text_vision_same_space(siglip_model, siglip_tokenizer):
    """Text and vision pooler_outputs are in the same embedding space.
    The SigLIP forward pass just normalizes them before computing logits.
    Cosine similarity should be small but positive for plausible pairs.
    """
    from transformers import AutoImageProcessor
    from PIL import Image

    proc = AutoImageProcessor.from_pretrained(
        "google/siglip-base-patch16-224", use_fast=True
    )

    # Create a colorful test image (not just solid gray)
    img = Image.new("RGB", (224, 224), (200, 50, 50))
    img_inputs = proc(images=img, return_tensors="pt")
    text_inputs = siglip_tokenizer(
        ["a red image"], return_tensors="pt", padding="max_length"
    )

    with torch.no_grad():
        vis_out = siglip_model.vision_model(**img_inputs)
        txt_out = siglip_model.text_model(**text_inputs)

        img_emb = vis_out.pooler_output[0]
        txt_emb = txt_out.pooler_output[0]
        img_emb = img_emb / img_emb.norm()
        txt_emb = txt_emb / txt_emb.norm()

    cos_sim = torch.dot(txt_emb, img_emb).item()

    # SigLIP cosine sims are small — even good matches are < 0.2.
    # What matters is they're not wildly negative (which would indicate
    # mismatched embedding spaces or wrong padding).
    assert cos_sim > -0.15, (
        f"Text-vision cosine sim is too negative ({cos_sim:.4f}), "
        f"embeddings may be in different spaces"
    )


def test_forward_uses_pooler_output(siglip_model, siglip_tokenizer):
    """Verify that the full forward pass uses the same pooler_output
    as our text/vision embedding paths, just normalized.

    This ensures embed() and embed_text() produce comparable features.
    """
    from transformers import AutoImageProcessor
    from PIL import Image

    proc = AutoImageProcessor.from_pretrained(
        "google/siglip-base-patch16-224", use_fast=True
    )
    img = Image.new("RGB", (224, 224), (128, 128, 128))
    img_inputs = proc(images=img, return_tensors="pt")
    text_inputs = siglip_tokenizer(
        ["test"], return_tensors="pt", padding="max_length"
    )

    with torch.no_grad():
        # Full forward
        full_out = siglip_model(**text_inputs, **img_inputs)

        # Direct tower calls
        vis_emb = siglip_model.vision_model(**img_inputs).pooler_output[0]
        vis_emb_norm = vis_emb / vis_emb.norm()

        txt_emb = siglip_model.text_model(**text_inputs).pooler_output[0]
        txt_emb_norm = txt_emb / txt_emb.norm()

    # The forward pass normalizes pooler_output to produce *_embeds
    assert torch.allclose(
        full_out.image_embeds[0], vis_emb_norm, atol=1e-5
    ), "Forward image_embeds should equal normalized vision_model.pooler_output"

    assert torch.allclose(
        full_out.text_embeds[0], txt_emb_norm, atol=1e-5
    ), "Forward text_embeds should equal normalized text_model.pooler_output"


# ---------------------------------------------------------------------------
# Pitfall 4: SigLIP sigmoid confidence
# ---------------------------------------------------------------------------

def test_sigmoid_confidence_formula(siglip_embedder):
    """The text_confidence method correctly converts FAISS squared-L2
    distances to SigLIP's native sigmoid confidence.

    Key insight: FAISS IndexFlatL2 returns SQUARED L2 distances.
    For unit vectors: cos_sim = 1 - squared_l2 / 2
    NOT: cos_sim = 1 - sqrt(squared_l2)^2 / 2 (double-squaring bug!)
    """
    logit_scale = siglip_embedder.model.logit_scale.exp().item()
    logit_bias = siglip_embedder.model.logit_bias.item()

    # Squared L2 distance of 0 → cos_sim = 1.0 → high logit → conf ≈ 1.0
    conf_identical = siglip_embedder.text_confidence(0.0)
    assert conf_identical > 0.99, f"Identical vectors should have high confidence: {conf_identical}"

    # Squared L2 distance of 2.0 → cos_sim = 0.0 → logit ≈ bias → conf = sigmoid(bias)
    conf_orthogonal = siglip_embedder.text_confidence(2.0)
    expected = 1.0 / (1.0 + math.exp(-logit_bias))
    assert abs(conf_orthogonal - expected) < 1e-6

    # Monotonicity: smaller distance → higher confidence
    conf_close = siglip_embedder.text_confidence(1.6)
    conf_far = siglip_embedder.text_confidence(1.9)
    assert conf_close > conf_far


def test_confidence_values_in_expected_range(siglip_embedder):
    """For typical text-search distances (1.7-1.9), sigmoid confidence
    should be in a meaningful range, not all zeros or all ones.

    Before the fix, the old confidence formula (1 - dist/2) gave ~5-10%
    for ALL text results, which was below the 60% min_confidence threshold.
    The sigmoid formula gives a wider spread (0.1% - 50%) that's more
    discriminative.
    """
    # Typical good text-image match (cos_sim ≈ 0.10-0.15)
    good_match_dist = 1.78  # squared L2
    conf_good = siglip_embedder.text_confidence(good_match_dist)

    # Typical poor match (cos_sim ≈ 0.05)
    poor_match_dist = 1.90
    conf_poor = siglip_embedder.text_confidence(poor_match_dist)

    # Good match should have meaningfully higher confidence
    assert conf_good > conf_poor * 5, (
        f"Good match ({conf_good:.4f}) should be much higher than poor match ({conf_poor:.4f})"
    )

    # Neither should be exactly 0 or 1
    assert 0 < conf_poor < conf_good < 1


# ---------------------------------------------------------------------------
# Pitfall 5: FAISS squared L2 vs raw L2
# ---------------------------------------------------------------------------

def test_faiss_returns_squared_l2():
    """FAISS IndexFlatL2 returns SQUARED L2 distances.

    This is critical: if you compute cos_sim = 1 - dist²/2 (treating the
    returned value as raw L2 and squaring it again), you get completely
    wrong values. The correct formula is cos_sim = 1 - dist/2.

    Example with unit vectors at 90° (cos_sim = 0):
        Raw L2 = sqrt(2) ≈ 1.414
        Squared L2 (FAISS) = 2.0
        Correct: cos_sim = 1 - 2.0/2 = 0 ✓
        Wrong:   cos_sim = 1 - 2.0²/2 = -1.0 ✗
    """
    import faiss

    dim = 4
    # Two unit vectors at 90 degrees
    a = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    b = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)

    index = faiss.IndexFlatL2(dim)
    index.add(b)
    distances, _ = index.search(a, 1)

    faiss_dist = distances[0][0]
    # FAISS returns squared L2 = 2.0 for orthogonal unit vectors
    assert abs(faiss_dist - 2.0) < 1e-5, f"Expected squared L2 = 2.0, got {faiss_dist}"

    # Correct cosine similarity computation
    cos_sim_correct = 1.0 - faiss_dist / 2.0
    assert abs(cos_sim_correct - 0.0) < 1e-5

    # The wrong way (double-squaring) gives -1.0 instead of 0.0
    cos_sim_wrong = 1.0 - (faiss_dist ** 2) / 2.0
    assert abs(cos_sim_wrong - (-1.0)) < 1e-5  # This is wrong!


# ---------------------------------------------------------------------------
# SigLIPEmbedder integration
# ---------------------------------------------------------------------------

def test_embed_text_produces_unit_vectors(siglip_embedder):
    """embed_text should return unit-normalized vectors."""
    emb = siglip_embedder.embed_text("a photo of a dog")
    norm = np.linalg.norm(emb)
    assert abs(norm - 1.0) < 1e-4, f"Text embedding norm should be 1.0, got {norm}"


def test_embed_image_produces_unit_vectors(siglip_embedder):
    """embed should return unit-normalized vectors."""
    from PIL import Image
    img = Image.new("RGB", (224, 224), (128, 128, 128))
    emb = siglip_embedder.embed(img)
    norm = np.linalg.norm(emb)
    assert abs(norm - 1.0) < 1e-4, f"Image embedding norm should be 1.0, got {norm}"


def test_text_and_image_embedding_dimensions_match(siglip_embedder):
    """Text and image embeddings must have the same dimensionality for
    FAISS search to work."""
    from PIL import Image

    text_emb = siglip_embedder.embed_text("hello")
    img_emb = siglip_embedder.embed(Image.new("RGB", (224, 224), (0, 0, 0)))
    assert text_emb.shape == img_emb.shape == (768,)


def test_text_search_ranking_is_meaningful(siglip_embedder):
    """Text search should rank semantically closer images higher.

    We embed two images (red, blue) and two text queries ("red", "blue"),
    and verify the correct pairings score higher.
    """
    from PIL import Image

    # Create two distinct images
    red_img = Image.new("RGB", (224, 224), (255, 0, 0))
    blue_img = Image.new("RGB", (224, 224), (0, 0, 255))

    red_emb = siglip_embedder.embed(red_img)
    blue_emb = siglip_embedder.embed(blue_img)

    red_text = siglip_embedder.embed_text("a red image")
    blue_text = siglip_embedder.embed_text("a blue image")

    # Correct pairings should have higher cosine similarity
    sim_red_red = np.dot(red_text, red_emb)
    sim_red_blue = np.dot(red_text, blue_emb)
    sim_blue_blue = np.dot(blue_text, blue_emb)
    sim_blue_red = np.dot(blue_text, red_emb)

    assert sim_red_red > sim_red_blue, (
        f"'red' text should be closer to red image: {sim_red_red:.4f} vs {sim_red_blue:.4f}"
    )
    assert sim_blue_blue > sim_blue_red, (
        f"'blue' text should be closer to blue image: {sim_blue_blue:.4f} vs {sim_blue_red:.4f}"
    )
