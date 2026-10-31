"""Use an optional Sol threshold during scheduled dense steps."""
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
from comfyui_turing_utils.attention.protocol import AttentionBackendCapabilities, AttentionExecutionOutcome
from comfyui_turing_utils.attention.sparse import SolAttentionCall
from comfyui_turing_utils.attention.stable import AttentionCall
from comfyui_turing_utils.nodes.attention import AttentionStrategy, _ATTENTION_STRATEGIES


class SolDenseStepTest(unittest.TestCase):
    def test_all_entry_points_select_dense_or_sol_for_prefix_and_suffix_steps(self):
        for path in ("tensor", "container", "prepared", "streamed"):
            for threshold in (0.0, 0.4):
                with self.subTest(path=path, threshold=threshold), ExitStack() as stack:
                    for name in ("is_supported_attention_device", "bundled_sparse_available",
                                 "bundled_w8a8_available", "split_prequantization_available",
                                 "fused_qk_preprocessing_available"):
                        stack.enter_context(mock.patch.object(sol, name, return_value=True))
                    stack.enter_context(mock.patch.object(sol, "preflight_bundled_sparse"))
                    stack.enter_context(mock.patch.object(
                        AttentionBackendCapabilities, "unsupported_reason", return_value=None))
                    stack.enter_context(mock.patch.object(sol, "_prepared_call_mismatch", return_value=None))
                    call = SolAttentionCall(AttentionCall(1, 1, 1, 64, 64, 64,
                                                         "HND", torch.float16, False),
                                            64, (), (), 1)
                    stack.enter_context(mock.patch.object(sol, "inspect_sol_attention_call",
                                                          return_value=(call, None)))
                    stack.enter_context(mock.patch.object(sol, "prequantize_turing_qk"))
                    quantize = stack.enter_context(mock.patch.object(
                        sol, "prequantize_turing_sol_attention" if path == "container"
                        else "prequantize_turing_sol_attention_from_qk"))
                    tensor_sparse = stack.enter_context(mock.patch.object(
                        sol, "turing_sol_sparse_attention", return_value="sparse"))
                    stack.enter_context(mock.patch.object(sol, "turing_sol_attention_from_prequantized",
                                                          return_value="sparse"))
                    dense = lambda *args, **kwargs: "dense"
                    dense.container_function = dense
                    dense.prepared_attention_executor = lambda request: AttentionExecutionOutcome("dense")
                    dense.prepared_attention_executor.turing_utils_streamed_qkv_executor = (
                        lambda *args, **kwargs: AttentionExecutionOutcome("dense"))
                    override = sol.make_sparse_attention_override(
                        torch.device("cuda"), dense_override=dense,
                        routing_threshold=1.3, dense_prefix_steps=1, dense_suffix_steps=1,
                        dense_prefix_layers=1, dense_step_threshold=threshold)
                    tensor = torch.zeros(1, 1, 64, 64, dtype=torch.float16)
                    for step, sigma in enumerate((1.0, 0.7, 0.3)):
                        for layer in (0, 3):
                            options = {"sample_sigmas": torch.tensor([1.0, 0.7, 0.3, 0.0]),
                                       "sigmas": torch.tensor([sigma]),
                                       "turing_utils_attention_layout": {"layer_index": layer, "layer_count": 10}}
                            container = lambda: SimpleNamespace(peek=lambda: tensor, take=lambda: tensor)
                            if path == "tensor":
                                result = override(None, transformer_options=options)
                            elif path == "container":
                                result = override.container_function(container(), container(), container(), 1,
                                                                     transformer_options=options)
                            elif path == "prepared":
                                result = override.prepared_attention_executor(SimpleNamespace(
                                    transformer_options=options, peek_qkv=lambda: (tensor,) * 3,
                                    consume_qkv=lambda: (tensor,) * 3, heads=1, kv_heads=1, mask=None,
                                    skip_output_reshape=False, low_precision_attention=True,
                                    is_causal=False, qk_transform=None, scale=None))
                                result = result.output
                            else:
                                result = override.prepared_attention_executor.turing_utils_streamed_qkv_executor(
                                    None, tensor, heads=1, qk_transform=None, transformer_options=options).output
                            expected_dense = layer == 0 or (threshold == 0 and step != 1)
                            self.assertEqual(result, "dense" if expected_dense else "sparse")
                            if not expected_dense:
                                selected = tensor_sparse if path == "tensor" else quantize
                                self.assertEqual(selected.call_args.kwargs["routing_threshold"],
                                                 1.3 if step == 1 else threshold)

    def test_node_places_one_numeric_control_below_dense_steps(self):
        strategy = next(item for item in AttentionStrategy.define_schema().inputs if item.id == "strategy")
        names = [item.id for item in strategy.options[0].inputs]
        self.assertEqual(names[names.index("dense_suffix_steps") + 1], "dense_step_threshold")
        field = next(item for item in strategy.options[0].inputs if item.id == "dense_step_threshold")
        self.assertEqual(field.default, 0.0)
        apply = mock.Mock(return_value=object())
        with mock.patch.dict(_ATTENTION_STRATEGIES, {"sol": apply}):
            AttentionStrategy.execute(object(), {"strategy": "sol", "dense_step_threshold": 0.4})
        self.assertEqual(apply.call_args.kwargs["dense_step_threshold"], 0.4)


if __name__ == "__main__":
    unittest.main()
