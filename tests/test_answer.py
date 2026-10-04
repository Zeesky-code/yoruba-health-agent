from agent.tools import Citation, parse_numbered, split_sentences


def test_split_sentences_attaches_overlapping_citations():
    text = "Malaria spreads through mosquito bites. Sleep under a net. See a doctor."
    cites = [
        Citation(start=0, end=7, text="Malaria", chunk_ids=["315-0"]),
        Citation(start=46, end=58, text="under a net", chunk_ids=["315-0", "6545-1"]),
    ]
    assert split_sentences(text, cites) == [
        ("Malaria spreads through mosquito bites.", ["315-0"]),
        ("Sleep under a net.", ["315-0", "6545-1"]),
        ("See a doctor.", []),
    ]


def test_parse_numbered_requires_every_line():
    assert parse_numbered("1. Ọ̀kan\n2) Èjì", 2) == ["Ọ̀kan", "Èjì"]
    assert parse_numbered("Here you go:\n1. Ọ̀kan\n2. Èjì", 2) == ["Ọ̀kan", "Èjì"]
    assert parse_numbered("1. Ọ̀kan", 2) is None


def test_split_sentences_handles_list_items():
    text = "Changes can help:\n- Losing weight\n- Being active\nSee a doctor."
    assert [s for s, _ in split_sentences(text, [])] == [
        "Changes can help:",
        "Losing weight",
        "Being active",
        "See a doctor.",
    ]
