"""Correctness tests for evaluate.py metrics (pure python, no models).

Run:  python -m unittest tests.test_metrics -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate as E  # noqa: E402

N = E.NormCfg()  # default normalization


class TestLevenshtein(unittest.TestCase):
    def test_known_values(self):
        self.assertEqual(E.levenshtein("kitten", "sitting"), 3)
        self.assertEqual(E.levenshtein("flaw", "lawn"), 2)
        self.assertEqual(E.levenshtein("", "abc"), 3)
        self.assertEqual(E.levenshtein("abc", "abc"), 0)

    def test_works_on_token_lists(self):
        self.assertEqual(E.levenshtein(["a", "b", "c"], ["a", "x", "c"]), 1)
        self.assertEqual(E.levenshtein(["a", "b"], ["a", "b", "c"]), 1)


class TestCerWer(unittest.TestCase):
    def test_perfect(self):
        e, n = E.cer_counts("hello world", "hello world", N)
        self.assertEqual((e, n), (0, 11))

    def test_one_char_sub(self):
        e, n = E.cer_counts("abcd", "abce", N)
        self.assertEqual(e, 1)
        self.assertAlmostEqual(e / n, 0.25)

    def test_wer(self):
        e, n = E.wer_counts("the quick brown fox", "the quick brown", N)
        self.assertEqual((e, n), (1, 4))  # one deletion out of 4 ref words

    def test_whitespace_collapsed_not_penalized(self):
        e, _ = E.cer_counts("a b c", "a   b\n c", N)
        self.assertEqual(e, 0)

    def test_case_and_punct_flags(self):
        raw = E.NormCfg()
        soft = E.NormCfg(lowercase=True, strip_punct=True)
        self.assertGreater(E.cer_counts("Hello, World", "hello world", raw)[0], 0)
        self.assertEqual(E.cer_counts("Hello, World", "hello world", soft)[0], 0)


class TestWordAccuracy(unittest.TestCase):
    def test_perfect_recovery(self):
        m, n = E.word_acc_counts("the cat sat", "the cat sat", N)
        self.assertEqual((m, n), (3, 3))

    def test_order_free(self):
        # fully reordered but every word present -> 100% recovery
        m, n = E.word_acc_counts("alpha beta gamma", "gamma alpha beta", N)
        self.assertEqual(m, 3)
        self.assertEqual(n, 3)

    def test_recognition_error_drops_one_word(self):
        m, n = E.word_acc_counts("alpha beta gamma", "alpha bete gamma", N)
        self.assertEqual(m, 2)               # 'beta' misread -> not matched
        self.assertEqual(n, 3)

    def test_multiset_not_set(self):
        # duplicate ref word only counts as recovered as many times as predicted
        m, n = E.word_acc_counts("the the cat", "the cat", N)
        self.assertEqual(m, 2)               # one 'the' + 'cat'
        self.assertEqual(n, 3)


class TestScoreAggregation(unittest.TestCase):
    def test_corpus_aggregation(self):
        sc = E.Score(system="x")
        sc.cer_edits, sc.cer_ref = 3, 100
        sc.wer_edits, sc.wer_ref = 2, 40
        sc.wa_matched, sc.wa_total = 36, 40
        self.assertAlmostEqual(sc.cer, 0.03)
        self.assertAlmostEqual(sc.wer, 0.05)
        self.assertAlmostEqual(sc.word_acc, 0.90)

    def test_no_word_data(self):
        sc = E.Score(system="x", cer_edits=1, cer_ref=10)
        self.assertIsNone(sc.word_acc)


class TestDatasetHelpers(unittest.TestCase):
    def test_auto_col_detection(self):
        row = {"image": 1, "text": "hi"}
        self.assertEqual(E._auto_col(row, E._IMG_COL_HINTS), "image")
        self.assertEqual(E._auto_col(row, E._TXT_COL_HINTS), "text")

    def test_auto_col_fuzzy_and_failure(self):
        row = {"jpg_bytes": 1, "transcription": "x"}
        self.assertEqual(E._auto_col(row, E._TXT_COL_HINTS), "transcription")
        with self.assertRaises(KeyError):
            E._auto_col({"foo": 1}, E._IMG_COL_HINTS)

    def test_sample_in_memory_image(self):
        import numpy as np
        arr = np.zeros((10, 20), np.uint8)              # grayscale -> RGB
        s = E.Sample(name="x", gt="hi", image=arr)
        out = s.image_rgb()
        self.assertEqual(out.shape, (10, 20, 3))

    def test_iam_preset_is_line_level(self):
        self.assertTrue(E.is_line_level("iam-lines"))
        self.assertFalse(E.is_line_level("iam-sentences"))   # page-level
        self.assertFalse(E.is_line_level("some/random-pages"))


def _word(text, x, y, line_idx):
    return {"text": text, "line_idx": line_idx,
            "polygon": {"x0": x, "y0": y, "x1": x + 10, "y1": y,
                        "x2": x + 10, "y2": y + 8, "x3": x, "y3": y + 8}}


class TestGNHK(unittest.TestCase):
    def test_reading_order_reconstruction(self):
        # deliberately out of order; correct reading order is by line then x
        words = [
            _word("world", 120, 10, 0), _word("Hello", 20, 12, 0),
            _word("line", 60, 60, 1), _word("second", 10, 58, 1),
        ]
        self.assertEqual(E._gnhk_reading_order_text(words), "Hello world\nsecond line")

    def test_special_tokens_dropped(self):
        words = [_word("real", 10, 10, 0), _word("%NA%", 80, 10, 0),
                 _word("%math%", 10, 50, 1)]
        self.assertEqual(E._gnhk_reading_order_text(words), "real")


class TestEvaluateSystemWordAcc(unittest.TestCase):
    def test_word_acc_order_free_but_cer_penalizes_order(self):
        import numpy as np
        s = E.Sample(name="x", gt="line one\nline two",
                     image=np.zeros((8, 8, 3), "uint8"))
        sc = E.evaluate_system("t", [s], predict_text=lambda im: "line two line one")
        self.assertAlmostEqual(sc.word_acc, 1.0, msg="all words present despite swap")
        self.assertGreater(sc.cer, 0, "document CER still penalizes the swap")

    def test_recognition_error_drops_word_acc(self):
        import numpy as np
        s = E.Sample(name="x", gt="hello world", image=np.zeros((8, 8, 3), "uint8"))
        sc = E.evaluate_system("t", [s], predict_text=lambda im: "hallo world")
        self.assertAlmostEqual(sc.word_acc, 0.5)   # 1 of 2 words correct


if __name__ == "__main__":
    unittest.main(verbosity=2)
