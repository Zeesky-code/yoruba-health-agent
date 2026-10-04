import numpy as np

from agent.tools import Chunk, top_k
from corpus.build_corpus import chunk_text, html_to_text, read_topics


def words(text: str) -> int:
    return len(text.split())


def test_html_to_text_keeps_paragraphs_and_list_items():
    html = "<p>Malaria is  serious.</p><ul><li>Wear repellent</li><li>Cover up</li></ul>"
    assert html_to_text(html) == "Malaria is serious.\n- Wear repellent\n- Cover up"


def test_short_text_is_one_chunk():
    assert chunk_text("One. Two.", words, max_tokens=10, overlap=2) == ["One. Two."]


def test_chunks_respect_limit_and_overlap():
    text = " ".join(f"Sentence number {i} here." for i in range(20))  # 4 words each
    chunks = chunk_text(text, words, max_tokens=12, overlap=4)
    assert len(chunks) > 1
    assert all(words(c) <= 12 for c in chunks)
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        last_sentence = prev.split(". ")[-1]
        assert nxt.startswith(last_sentence)


def test_chunking_preserves_paragraph_breaks():
    text = "First para. Still first.\n- A list item"
    assert chunk_text(text, words, max_tokens=50, overlap=5) == [text]


def test_read_topics_maps_titles_to_areas(tmp_path):
    f = tmp_path / "topics.txt"
    f.write_text("# comment\n[malaria]\nMalaria\nFever\n\n[tb]\nTuberculosis\n")
    assert read_topics(f) == {"Malaria": "malaria", "Fever": "malaria", "Tuberculosis": "tb"}


def test_top_k_orders_by_cosine():
    chunks = [Chunk(id=str(i), topic="t", area="a", url="u", text="x") for i in range(3)]
    matrix = np.eye(3, dtype=np.float32)
    hits = top_k(np.array([0.1, 0.9, 0.3], dtype=np.float32), chunks, matrix, k=2)
    assert [h.id for h in hits] == ["1", "2"]
    assert hits[0].score > hits[1].score
