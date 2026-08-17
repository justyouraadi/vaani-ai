from vaani.tts.chunker import SentenceChunker


def test_first_token_flushes_immediately():
    c = SentenceChunker(240)
    out = c.push("Hmm, ")
    assert out == [("Hmm, ", False)]


def test_punctuation_splits_sentences():
    c = SentenceChunker(240)
    assert c.push("Hmm, ") == [("Hmm, ", False)]
    assert c.push("namaste ") == []
    assert c.push("main Vaani ") == []
    assert c.push("hoon!") == [("namaste main Vaani hoon!", True)]


def test_max_chars_hard_split_at_space():
    c = SentenceChunker(20)
    c.push("short")
    out = c.push("word" * 30)
    assert out
    for text, _ in out:
        assert len(text) <= 20


def test_flush_emits_remainder():
    c = SentenceChunker(240)
    c.push("Ji, ")
    c.push("aaj kaafi ")
    out = c.flush()
    assert out == [("aaj kaafi", True)]
    assert c.flush() == []


def test_danda_splits():
    c = SentenceChunker(240)
    c.push("Hmm, ")
    assert c.push("kaise ho bhaiya।") == [("kaise ho bhaiya।", True)]