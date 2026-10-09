from ingest import chunk_text, split_fixed, split_title


def test_each_paragraph_becomes_a_chunk():
    assert chunk_text("First para.\n\nSecond para.") == ["First para.", "Second para."]


def test_empty_pieces_are_skipped():
    assert chunk_text("\n\nA\n\n\n\n\nB\n\n") == ["A", "B"]


def test_long_paragraph_falls_back_to_fixed_windows():
    chunks = chunk_text("x" * 1200, size=500, overlap=100)
    assert [len(c) for c in chunks] == [500, 500, 400]


def test_windows_overlap():
    pieces = split_fixed("abcdefghij", size=4, overlap=2)
    assert pieces[:2] == ["abcd", "cdef"]


def test_split_title_separates_title_block():
    title, body = split_title("My Title\n(Source: x)\n\nBody para.")
    assert title == "My Title"
    assert body == "Body para."