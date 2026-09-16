import unicodedata

from leanvocab.data.normalize import nfc


def test_decomposed_accent_is_composed() -> None:
    assert nfc("Café") == "Café"


def test_nfc_is_idempotent() -> None:
    s = "Å ﬁ ẛ̣"
    assert nfc(nfc(s)) == nfc(s)
    assert unicodedata.is_normalized("NFC", nfc(s))


def test_nfc_does_not_apply_compatibility_folding() -> None:
    assert nfc("ﬁ") == "ﬁ"
