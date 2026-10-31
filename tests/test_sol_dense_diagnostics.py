"""Scheduled dense attention must be visible on every Sol entry point."""
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parents[1])]
from comfyui_turing_utils.attention import sol
from comfyui_turing_utils.attention.protocol import AttentionBackendCapabilities


class SolDenseDiagnosticsTest(unittest.TestCase):
    def test_dense_prefix_is_reported_without_flash_on_all_entry_points(self):
        for path in ("tensor", "container", "prepared", "streamed"):
            with self.subTest(path=path), ExitStack() as stack:
                for name in (
                    "is_supported_attention_device", "bundled_sparse_available",
                    "bundled_w8a8_available", "split_prequantization_available",
                    "fused_qk_preprocessing_available",
                ):
                    stack.enter_context(mock.patch.object(sol, name, return_value=True))
                stack.enter_context(mock.patch.object(sol, "preflight_bundled_sparse"))
                stack.enter_context(mock.patch.object(
                    AttentionBackendCapabilities, "unsupported_reason", return_value=None
                ))
                log = stack.enter_context(mock.patch.object(sol.LOG, "info"))
                dense = lambda *args, **kwargs: "dense"
                dense.container_function = dense
                dense.prepared_attention_executor = lambda request: "dense"
                dense.prepared_attention_executor.turing_utils_streamed_qkv_executor = dense
                override = sol.make_sparse_attention_override(
                    torch.device("cuda"), dense_override=dense, dense_backend="w8a8",
                    dense_prefix_steps=1, dense_prefix_layers=0, debug_route_density=True,
                )
                options = {"sample_sigmas": torch.tensor([1.0, 0.5, 0.0]),
                           "sigmas": torch.tensor([1.0])}
                if path == "tensor":
                    result = override(None, transformer_options=options)
                elif path == "container":
                    result = override.container_function(None, None, None, 1,
                                                         transformer_options=options)
                elif path == "prepared":
                    result = override.prepared_attention_executor(
                        SimpleNamespace(transformer_options=options))
                else:
                    result = override.prepared_attention_executor.turing_utils_streamed_qkv_executor(
                        None, torch.zeros(1, 1, 64, 64), heads=1,
                        qk_transform=None, transformer_options=options)
                self.assertEqual(result, "dense")
                self.assertTrue(any("selected by schedule" in call.args[0]
                                    for call in log.call_args_list))
                self.assertEqual(log.call_args.args[1:3], (0, 2))


if __name__ == "__main__":
    unittest.main()
