import os
import tempfile
import unittest

import numpy as np

try:
    import onnx
    from onnx import TensorProto, helper
except ImportError:      # the plumbing test needs the onnx package to build a tiny stand-in model
    onnx = None

from detector.audio import AudioListener, HornDetector, HornEvent


def tiny_model(path):
    """A stand-in for YAMNet: input 15600 samples, output 1 x 521 scores. 'Horn' score rises with loudness."""
    w = np.zeros((1, 521), np.float32)
    w[0, 302] = 40.0
    w[0, 6] = -40.0
    b = np.full((1, 521), -10.0, np.float32)
    nodes = [
        helper.make_node("Abs", ["x"], ["a"]),
        helper.make_node("ReduceMean", ["a"], ["m"], keepdims=1),
        helper.make_node("Reshape", ["m", "shape"], ["m2"]),
        helper.make_node("Mul", ["m2", "w"], ["p"]),
        helper.make_node("Add", ["p", "b"], ["z"]),
        helper.make_node("Sigmoid", ["z"], ["y"]),
    ] 
    g = helper.make_graph(
        nodes, "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [15600])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 521])],
        [helper.make_tensor("shape", TensorProto.INT64, [2], [1, 1]),
         helper.make_tensor("w", TensorProto.FLOAT, [1, 521], w.flatten().tolist()),
         helper.make_tensor("b", TensorProto.FLOAT, [1, 521], b.flatten().tolist())])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, path)


class HornModelTests(unittest.TestCase):
    @unittest.skipIf(onnx is None, "onnx package not installed")
    def test_classifier_and_listener_filter(self):
        from detector.horn_classifier import HornClassifier
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.onnx")
            tiny_model(path)
            clf = HornClassifier(path)
            t = np.arange(16000) / 16000.0
            loud = (0.9 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            quiet = (0.001 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            self.assertGreater(clf.judge(loud, 16000).horn, 0.5)
            self.assertLess(clf.judge(quiet, 16000).horn, 0.05)
            self.assertGreater(clf.judge(loud[:6000], 16000).horn, 0.1)      # a short sound is padded and still scored
            # 8 kHz input is resampled to 16 kHz
            self.assertGreater(clf.judge(loud[::2], 8000).horn, 0.5)

            # the listener keeps events the model likes and drops the others (no microphone needed)
            lst = AudioListener.__new__(AudioListener)
            import collections
            import queue
            lst.detector = type("D", (), {"sr": 16000})()
            lst.classifier, lst.min_score = clf, 0.3
            lst._ring, lst._pending, lst.events = collections.deque(), [], queue.Queue()
            lst._ring.append((1000.0, np.concatenate([quiet, loud, loud])))
            good = HornEvent(start=1001.0, end=1001.5, duration_s=0.5, peak_hz=440, level_db=-5, excess_db=20, tonal=0.9)
            bad = HornEvent(start=1000.0, end=1000.5, duration_s=0.5, peak_hz=440, level_db=-5, excess_db=20, tonal=0.9)
            lst._pending = [good, bad]
            lst._judge_pending(1003.0)
            kept = [lst.events.get_nowait() for _ in range(lst.events.qsize())]
            self.assertEqual(len(kept), 1)
            self.assertEqual(kept[0].start, 1001.0)
            self.assertGreater(kept[0].classifier_score, 0.3)
            self.assertIn("Vehicle horn", kept[0].sound_note)      # the administrator can see what was heard

    def test_judgement_rules(self):
        from detector.horn_classifier import HornClassifier, Judgement
        self.assertTrue(Judgement(0.8, 0.1, 0.1).accepted(0.3))
        self.assertFalse(Judgement(0.2, 0.0, 0.0).accepted(0.3))     # not sure enough
        self.assertFalse(Judgement(0.5, 0.7, 0.0).accepted(0.3))     # someone shouting is stronger
        self.assertFalse(Judgement(0.5, 0.0, 0.9).accepted(0.3))     # a siren is stronger

        # Even with a fine-tuned head loaded, judge() must populate voice & alarm scores so shouting/sirens are vetoed
        clf = HornClassifier.__new__(HornClassifier)
        clf.names = {302: "Vehicle horn", 6: "Shout", 317: "Siren"}
        clf.head = {
            "w": np.zeros(521, dtype=np.float64),
            "b": 2.0,   # sigmoid(2.0) ~= 0.88
            "mean": np.zeros(521, dtype=np.float64),
            "std": np.ones(521, dtype=np.float64),
        }
        fake_scores = np.full(521, 0.01, dtype=np.float32)
        fake_scores[6] = 0.95   # very strong Shout
        clf.features = lambda samples, sr: fake_scores
        verdict = clf.judge(np.zeros(16000, dtype=np.float32), 16000)
        self.assertAlmostEqual(verdict.voice, 0.95, places=2)
        self.assertFalse(verdict.accepted(0.8))


if __name__ == "__main__":
    unittest.main()
