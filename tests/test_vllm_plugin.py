from __future__ import annotations

import os
import sys
import types

import pytest
import torch
from compressed_tensors.compressors.pack_quantized.helpers import (
    pack_to_int32,
    unpack_from_int32,
)
from compressed_tensors.transform import TransformLocation

from warp_quant.hadamard_gptq import _hadamard_rotate_weight
from warp_quant.streaming_moe import dequantize_ple_rows_lut4
from warp_quant.vllm_plugin import (
    CustomLut4PLEMethod,
    _extract_w2_code_offset,
    _patch_ct_linear_transform,
    _patch_qwen4_exp_ple_lut4,
    fast_block_hadamard,
    register,
    repack_uint2b2_to_uint4b8,
    unpack_dequant_uint2b2,
)


def test_repack_uint2b2_to_uint4b8_2d_and_3d() -> None:
    torch.manual_seed(42)
    for shape in ((64, 128), (4, 32, 64)):
        q_int = torch.randint(-2, 2, shape, dtype=torch.int8)
        w2_packed = pack_to_int32(q_int, num_bits=2)
        w4_expected = pack_to_int32(q_int, num_bits=4)
        w4_actual = repack_uint2b2_to_uint4b8(w2_packed)
        assert w4_actual.shape == w4_expected.shape
        assert torch.equal(w4_actual, w4_expected)
        unpacked = unpack_from_int32(w4_actual, 4, torch.Size(shape))
        assert torch.equal(unpacked.to(torch.int8), q_int)


def test_repack_uint2b2_to_uint4b8_code_offset_half_bit_exact() -> None:
    torch.manual_seed(123)
    for shape in ((64, 128), (4, 32, 64)):
        q_int = torch.randint(-2, 2, shape, dtype=torch.int8)
        w2_packed = pack_to_int32(q_int, num_bits=2)
        w4_sym = repack_uint2b2_to_uint4b8(w2_packed, code_offset=0.5)
        unpacked_signed = unpack_from_int32(w4_sym, 4, torch.Size(shape)).to(torch.int8)
        # (u2 - 1.5) * s == (2*u2 - 3) * (s / 2) == (2*q_int + 1) * (s / 2)
        expected_signed = (2 * q_int + 1).to(torch.int8)
        assert torch.equal(unpacked_signed, expected_signed)

        scale_shape = (*shape[:-1], shape[-1] // 64)
        scale = (torch.randn(scale_shape, dtype=torch.bfloat16).abs() + 0.05).contiguous()
        dq_from_w4 = unpacked_signed.to(torch.bfloat16) * (scale * 0.5).repeat_interleave(64, dim=-1)
        dq_ref = (q_int.float() + 0.5).to(torch.bfloat16) * scale.repeat_interleave(64, dim=-1)
        assert torch.equal(dq_from_w4, dq_ref)

    with pytest.raises(ValueError, match="Unsupported code_offset"):
        repack_uint2b2_to_uint4b8(torch.zeros((8, 4), dtype=torch.int32), code_offset=0.25)


def test_unpack_dequant_uint2b2_matches_reference() -> None:
    torch.manual_seed(7)
    n, k, g = 32, 128, 64
    q_int = torch.randint(-2, 2, (n, k), dtype=torch.int8)
    scale = torch.randn(n, k // g, dtype=torch.bfloat16).abs() + 0.01
    w2_packed = pack_to_int32(q_int, num_bits=2)

    actual = unpack_dequant_uint2b2(w2_packed, scale, group_size=g, out_dtype=torch.bfloat16)
    expected = q_int.to(torch.bfloat16) * scale.repeat_interleave(g, dim=-1)
    assert torch.equal(actual, expected)

    actual_sym = unpack_dequant_uint2b2(
        w2_packed, scale, group_size=g, out_dtype=torch.bfloat16, code_offset=0.5
    )
    expected_sym = (q_int.float() + 0.5).to(torch.bfloat16) * scale.repeat_interleave(g, dim=-1)
    assert torch.equal(actual_sym, expected_sym)

    with pytest.raises(ValueError, match="Unsupported code_offset"):
        unpack_dequant_uint2b2(w2_packed, scale, group_size=g, code_offset=0.25)


def test_extract_w2_code_offset_validation() -> None:
    assert _extract_w2_code_offset(types.SimpleNamespace()) == 0.0
    assert _extract_w2_code_offset(types.SimpleNamespace(observer_kwargs={})) == 0.0
    assert (
        _extract_w2_code_offset(
            types.SimpleNamespace(config=types.SimpleNamespace(observer_kwargs={"code_offset": 0.5}))
        )
        == 0.5
    )
    assert (
        _extract_w2_code_offset(
            types.SimpleNamespace(weight_quant=types.SimpleNamespace(observer_kwargs={"code_offset": 0.5}))
        )
        == 0.5
    )
    for bad in (0.25, -0.5, 2.0, float("nan")):
        with pytest.raises(ValueError, match="Invalid W2 code_offset"):
            _extract_w2_code_offset(types.SimpleNamespace(observer_kwargs={"code_offset": bad}))


def test_fast_block_hadamard_validation_and_hadacore_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(11)
    x = torch.randn(5, 192, dtype=torch.bfloat16)
    assert torch.equal(fast_block_hadamard(x, block_size=1), x)

    y = fast_block_hadamard(x, block_size=64)
    expected = _hadamard_rotate_weight(x, block_size=64)
    assert torch.equal(y, expected)
    x_roundtrip = fast_block_hadamard(y, block_size=64)
    assert torch.allclose(x_roundtrip.float(), x.float(), atol=2e-2, rtol=2e-2)

    with pytest.raises(ValueError, match="power of 2"):
        fast_block_hadamard(x, block_size=48)
    with pytest.raises(ValueError, match="divisible by block_size"):
        fast_block_hadamard(torch.randn(2, 100), block_size=64)
    with pytest.raises(ValueError, match=">= 1"):
        fast_block_hadamard(x, block_size=0)

    calls: list[tuple[tuple[int, ...], bool]] = []
    fake_ops = types.ModuleType("vllm._custom_ops")

    def _fake_hadacore(t: torch.Tensor, inplace: bool = False) -> torch.Tensor:
        calls.append((tuple(t.shape), inplace))
        return t.mul_(2.0)

    fake_ops.hadacore_transform = _fake_hadacore  # type: ignore[attr-defined]
    fake_vllm = types.ModuleType("vllm")
    fake_vllm._custom_ops = fake_ops  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm._custom_ops", fake_ops)

    x_orig = x.clone()
    out = fast_block_hadamard(x, block_size=64)
    assert calls == [((5, 3, 64), True)]
    assert torch.equal(x, x_orig)
    assert torch.equal(out, x_orig * 2.0)


def test_custom_lut4_ple_method_storage_dequant_and_tp_sharding() -> None:
    torch.manual_seed(19)
    num_rows, dim, group_size = 64, 160, 32
    num_groups = dim // group_size
    codes = torch.randint(0, 16, (num_rows, dim), dtype=torch.uint8)
    w_packed = ((codes[:, 0::2] & 0x0F) | ((codes[:, 1::2] & 0x0F) << 4)).contiguous()
    w_scale = (torch.randn(num_rows, num_groups, dtype=torch.float16).abs() + 0.01).contiguous()
    w_lut = torch.linspace(-1.0, 1.0, 16, dtype=torch.bfloat16)

    method = CustomLut4PLEMethod(group_size=group_size)
    shape, dtype = method.get_storage_Spec(num_rows, dim, torch.bfloat16)
    assert shape == (num_rows, 90)
    assert dtype == torch.uint8

    with pytest.raises(ValueError, match="missing required"):
        method.pack_shard_weights(0, {"weight_packed": w_packed}, dim)

    combined_rows = method.pack_shard_weights(
        shard_idx=0,
        parts={"weight_packed": w_packed, "weight_scale": w_scale, "weight_lut": w_lut},
        embedding_dim_per_shard=dim,
    )
    assert combined_rows.shape == (num_rows, 90)

    token_ids = torch.tensor([[0, 5, 17, 63], [2, 19, 33, 40]], dtype=torch.long)
    expected = dequantize_ple_rows_lut4(
        w_packed[token_ids],
        w_scale[token_ids],
        w_lut,
        group_size=group_size,
        out_dtype=torch.bfloat16,
    )
    gathered = combined_rows.index_select(0, token_ids.reshape(-1)).reshape(*token_ids.shape, 90)
    actual = method.dequantize(gathered, dim, torch.bfloat16, shard_idx=0)
    assert torch.equal(actual, expected)

    with pytest.raises(RuntimeError, match="Missing Custom-LUT4 codebook"):
        method.dequantize(gathered, dim, torch.bfloat16, shard_idx=3)

    # Verify TP=2 sharding where dim=160 splits into 80 columns per rank (g_eff = gcd(32, 80) = 16)
    rank_pieces = []
    for tp_rank in (0, 1):
        m_tp = CustomLut4PLEMethod(group_size=group_size)
        tp_shape, _ = m_tp.get_storage_Spec(num_rows, dim // 2, torch.bfloat16)
        assert tp_shape == (num_rows, 40 + (80 // 16) * 2)
        packed_rank = m_tp.pack_shard_weights(
            shard_idx=0,
            parts={"weight_packed": w_packed, "weight_scale": w_scale, "weight_lut": w_lut},
            embedding_dim_per_shard=dim // 2,
            tp_size=2,
            tp_rank=tp_rank,
        )
        g_rank = packed_rank.index_select(0, token_ids.reshape(-1)).reshape(*token_ids.shape, -1)
        rank_pieces.append(m_tp.dequantize(g_rank, dim // 2, torch.bfloat16, shard_idx=0))
    assert torch.equal(torch.cat(rank_pieces, dim=-1), expected)


def test_patch_ct_linear_transform_and_ple_with_stub_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeQuantMethod:
        def apply(
            self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
        ) -> torch.Tensor:
            del layer
            return x + bias if bias is not None else x

    class FakeLinearTransformMethod:
        def __init__(self) -> None:
            self.input_tfms = {0: "scheme_a"}
            self.output_tfms: dict[int, str] = {}
            self.input_transform = lambda t: t * 2.0
            self.output_transform = None
            self.quant_method = FakeQuantMethod()

        def _validate_tfm_schemes(self, num_partitions: int) -> None:
            for i in range(num_partitions):
                _ = self.input_tfms[i]

        def apply(
            self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
        ) -> torch.Tensor:
            assert bias is None
            return x

    class FakeHadamardTransform:
        def __init__(
            self,
            transforms: dict[int, object],
            layer: torch.nn.Module,
            weight_loader: object,
            input_size_per_partition: int,
            output_partition_sizes: list[int],
        ) -> None:
            del transforms, layer, weight_loader, input_size_per_partition, output_partition_sizes
            if fake_mod.get_tensor_model_parallel_world_size() > 1:
                raise NotImplementedError("Online transforms with tensor parallelism is not supported")

    fake_dist_par = types.ModuleType("vllm.distributed.parallel_state")
    fake_dist_par.get_tensor_model_parallel_world_size = lambda: 2  # type: ignore[attr-defined]
    fake_lin = types.ModuleType("vllm.model_executor.layers.quantization.compressed_tensors.transform.linear")
    fake_lin.CompressedTensorsLinearTransformMethod = FakeLinearTransformMethod  # type: ignore[attr-defined]
    fake_mod = types.ModuleType("vllm.model_executor.layers.quantization.compressed_tensors.transform.module")
    fake_mod.HadamardTransform = FakeHadamardTransform  # type: ignore[attr-defined]
    fake_mod.get_tensor_model_parallel_world_size = fake_dist_par.get_tensor_model_parallel_world_size  # type: ignore[attr-defined]

    for name, mod in {
        "vllm": types.ModuleType("vllm"),
        "vllm.distributed": types.ModuleType("vllm.distributed"),
        "vllm.distributed.parallel_state": fake_dist_par,
        "vllm.model_executor": types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.layers": types.ModuleType("vllm.model_executor.layers"),
        "vllm.model_executor.layers.quantization": types.ModuleType(
            "vllm.model_executor.layers.quantization"
        ),
        "vllm.model_executor.layers.quantization.compressed_tensors": types.ModuleType(
            "vllm.model_executor.layers.quantization.compressed_tensors"
        ),
        "vllm.model_executor.layers.quantization.compressed_tensors.transform": types.ModuleType(
            "vllm.model_executor.layers.quantization.compressed_tensors.transform"
        ),
        "vllm.model_executor.layers.quantization.compressed_tensors.transform.linear": fake_lin,
        "vllm.model_executor.layers.quantization.compressed_tensors.transform.module": fake_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    _patch_ct_linear_transform()

    ltm = FakeLinearTransformMethod()
    # Unfused QKV (num_partitions=3, input_tfms only has key 0) should not raise KeyError
    ltm._validate_tfm_schemes(3)
    x = torch.ones(2, 4)
    bias = torch.full((4,), 0.5)
    out = ltm.apply(torch.nn.Identity(), x, bias=bias)
    assert torch.equal(out, x * 2.0 + bias)

    # Block-Hadamard TP=2 should succeed when head_dim divides input_size_per_partition
    fake_tfm = types.SimpleNamespace(
        scheme=types.SimpleNamespace(head_dim=64),
        args=types.SimpleNamespace(location=TransformLocation.INPUT),
    )
    FakeHadamardTransform({0: fake_tfm}, torch.nn.Identity(), lambda *a: None, 128, [128])


def test_patch_qwen4_exp_ple_lut4_end_to_end_with_stub_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePLEMethod:
        @classmethod
        def from_quant_config(cls, quant_config: object, hf_text_config: object = None) -> FakePLEMethod:
            del quant_config, hf_text_config
            return cls()

    class FakeDeviceEmbedding:
        def __init__(self, quant_method: FakePLEMethod, dim_per_shard: int) -> None:
            self.quant_method = quant_method
            self.embedding_dim_per_shard = dim_per_shard
            self.params_dtype = torch.bfloat16
            self.weights: dict[int, torch.Tensor] = {}

        def load_shard_weight(self, shard_idx: int, weight: torch.Tensor) -> None:
            self.weights[shard_idx] = weight

        def forward_SingleShard(
            self, shard_idx: int, flat_ids: torch.Tensor, valid_mask: torch.Tensor
        ) -> torch.Tensor:
            del shard_idx, flat_ids, valid_mask
            raise AssertionError("unpatched forward_SingleShard should not be called for LUT4")

    class FakePinnedHostEmbedding(FakeDeviceEmbedding):
        def __init__(self, quant_method: FakePLEMethod, dim_per_shard: int) -> None:
            super().__init__(quant_method, dim_per_shard)
            self._prefetched_out = None

        def _gather_rows_uva(self, shard_idx: int, flat_ids: torch.Tensor) -> torch.Tensor:
            return self.weights[shard_idx].index_select(0, flat_ids)

    class FakeNGramEmbedding:
        def __init__(self, backend: FakeDeviceEmbedding, num_shards: int, dim_per_shard: int) -> None:
            self.backend = backend
            self.num_shards = num_shards
            self.embedding_dim_per_shard = dim_per_shard
            self.tp_size = 1
            self.tp_rank = 0

        def load_weights(self, weights: object) -> set[str]:
            del weights
            return set()

    common_mod = types.ModuleType("vllm.models.qwen4_exp.common.ngram_embedding")
    common_mod.Qwen4ExpPLEEmbeddingMethod = FakePLEMethod  # type: ignore[attr-defined]
    common_mod.Qwen4ExpPLEDeviceEmbedding = FakeDeviceEmbedding  # type: ignore[attr-defined]
    common_mod.Qwen4ExpPLEPinnedHostEmbedding = FakePinnedHostEmbedding  # type: ignore[attr-defined]
    nvidia_mod = types.ModuleType("vllm.models.qwen4_exp.nvidia.ngram_embedding")
    nvidia_mod.Qwen4ExpNGramEmbedding = FakeNGramEmbedding  # type: ignore[attr-defined]

    for name, mod in {
        "vllm": types.ModuleType("vllm"),
        "vllm.models": types.ModuleType("vllm.models"),
        "vllm.models.qwen4_exp": types.ModuleType("vllm.models.qwen4_exp"),
        "vllm.models.qwen4_exp.common": types.ModuleType("vllm.models.qwen4_exp.common"),
        "vllm.models.qwen4_exp.common.ngram_embedding": common_mod,
        "vllm.models.qwen4_exp.nvidia": types.ModuleType("vllm.models.qwen4_exp.nvidia"),
        "vllm.models.qwen4_exp.nvidia.ngram_embedding": nvidia_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    _patch_qwen4_exp_ple_lut4()

    hf_cfg = types.SimpleNamespace(ple_quantization={"format": "custom_lut4_g32_fp16scale", "group_size": 32})
    qmethod = FakePLEMethod.from_quant_config(None, hf_cfg)
    assert isinstance(qmethod, CustomLut4PLEMethod)

    qcfg_lut4 = types.SimpleNamespace(ple_quantization={"format": "lut4-gptq", "group_size": 32})
    qmethod_from_qcfg = FakePLEMethod.from_quant_config(qcfg_lut4, types.SimpleNamespace())
    assert isinstance(qmethod_from_qcfg, CustomLut4PLEMethod)

    backend = FakePinnedHostEmbedding(qmethod, 160)
    ngram = FakeNGramEmbedding(backend, num_shards=1, dim_per_shard=160)

    w_packed = torch.randint(0, 256, (16, 80), dtype=torch.uint8)
    w_scale = (torch.randn(16, 5, dtype=torch.float16).abs() + 0.05).contiguous()
    w_lut = torch.linspace(-0.5, 0.5, 16, dtype=torch.bfloat16)
    w_shape = torch.tensor([16, 160], dtype=torch.int64)

    loaded = ngram.load_weights(
        [
            ("shard_0.weight_packed", w_packed),
            ("shard_0.weight_scale", w_scale),
            ("shard_0.weight_lut", w_lut),
            ("shard_0.weight_shape", w_shape),
        ]
    )
    assert loaded == {
        "shard_0.weight_packed",
        "shard_0.weight_scale",
        "shard_0.weight_lut",
        "shard_0.weight_shape",
    }
    flat_ids = torch.tensor([1, 4, 7], dtype=torch.long)
    valid_mask = torch.tensor([True, False, True])
    out = backend.forward_SingleShard(0, flat_ids, valid_mask)
    assert out.shape == (3, 160)
    assert torch.all(out[1] == 0)


def test_register_is_idempotent_without_vllm() -> None:
    register()
    register()


def test_hf_auto_map_warp_quant_config_from_checkpoint_dir(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import shutil
    from pathlib import Path

    from transformers import AutoConfig
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig

    import warp_quant.vllm_plugin as vp

    ckpt_dir = Path(str(tmp_path))
    shutil.copyfile(Path(vp.__file__).resolve(), ckpt_dir / "vllm_warp_quant.py")
    (ckpt_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "auto_map": {"AutoConfig": "vllm_warp_quant.WarpQuantConfig"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("WARP_QUANT_SUBPROCESS_AUTOLOAD", raising=False)

    cfg_trusted = AutoConfig.from_pretrained(str(ckpt_dir), trust_remote_code=True)
    assert isinstance(cfg_trusted, Qwen3_5MoeConfig)
    assert os.environ.get("WARP_QUANT_SUBPROCESS_AUTOLOAD") == "1"

    cfg_untrusted = AutoConfig.from_pretrained(str(ckpt_dir), trust_remote_code=False)
    assert isinstance(cfg_untrusted, Qwen3_5MoeConfig)


def test_hadamard_transform_forward_clones_input_and_humming_named_params_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import types

    par_mod = types.ModuleType("vllm.distributed.parallel_state")
    par_mod.get_tensor_model_parallel_world_size = lambda: 1
    ct_mod = types.ModuleType("vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors")
    lin_mod = types.ModuleType("vllm.model_executor.layers.quantization.compressed_tensors.transform.linear")
    ht_mod = types.ModuleType("vllm.model_executor.layers.quantization.compressed_tensors.transform.module")

    class FakeLinearTransformMethod:
        pass

    class FakeHadamardTransform(torch.nn.Module):
        def __init__(
            self, transforms, layer, weight_loader, input_size_per_partition, output_partition_sizes
        ):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(4), requires_grad=False)
            self.weight.partitions = {0: self.weight}
            self.transforms = {
                0: types.SimpleNamespace(
                    scheme=types.SimpleNamespace(type="hadamard", head_dim=4),
                    args=types.SimpleNamespace(location="input"),
                )
            }

        def forward(self, value: torch.Tensor, part_id: int = 0) -> torch.Tensor:
            # Simulate vLLM's in-place ops.hadacore_transform mutation
            value.mul_(2.0)
            return value

    lin_mod.CompressedTensorsLinearTransformMethod = FakeLinearTransformMethod
    ht_mod.HadamardTransform = FakeHadamardTransform

    monkeypatch.setitem(sys.modules, "vllm.distributed.parallel_state", par_mod)
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors",
        ct_mod,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.compressed_tensors.transform.linear",
        lin_mod,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.compressed_tensors.transform.module",
        ht_mod,
    )

    _patch_ct_linear_transform()

    ht = FakeHadamardTransform({}, torch.nn.Linear(4, 4), None, 4, [4])
    x = torch.ones(2, 4)
    out = ht(x)
    # Caller's x must NOT be mutated in-place
    assert torch.all(x == 1.0)
    assert torch.all(out == 2.0)


def test_hadamard_transform_tp2_shared_weight_and_straddling_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import types

    from compressed_tensors.transform import TransformLocation

    par_mod = types.ModuleType("vllm.distributed.parallel_state")
    par_mod.get_tensor_model_parallel_world_size = lambda: 2
    param_mod = types.ModuleType("vllm.model_executor.parameter")
    param_mod.get_tensor_model_parallel_world_size = lambda: 2
    comm_mod = types.ModuleType("vllm.distributed.communication_op")
    ct_mod = types.ModuleType("vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors")
    lin_mod = types.ModuleType("vllm.model_executor.layers.quantization.compressed_tensors.transform.linear")
    ht_mod = types.ModuleType("vllm.model_executor.layers.quantization.compressed_tensors.transform.module")
    ht_mod.get_tensor_model_parallel_world_size = lambda: 2

    class FakePartition(torch.nn.Parameter):
        def __new__(cls, data: torch.Tensor) -> FakePartition:
            return super().__new__(cls, data, requires_grad=False)

        def __init__(self, data: torch.Tensor) -> None:
            self.tp_rank = 1
            self.tp_size = 2

    class FakeSharedWeightParameter(torch.nn.Parameter):
        def __new__(cls, head_dim: int = 16) -> FakeSharedWeightParameter:
            if param_mod.get_tensor_model_parallel_world_size() > 1:
                raise NotImplementedError(
                    "SharedWeightParameter does not currently support tensor parallelism"
                )
            return super().__new__(cls, torch.empty(0), requires_grad=False)

        def __init__(self, head_dim: int = 16) -> None:
            self.partitions = {0: FakePartition(torch.empty((head_dim, head_dim)))}

        def load_row_parallel_weight(self, loaded_weight: torch.Tensor) -> None:
            part = self.partitions[0]
            assert part.tp_rank == 0, f"expected tp_rank=0 for replicated Hadamard matrix, got {part.tp_rank}"
            part.data.copy_(loaded_weight)

        def load_column_parallel_weight(self, loaded_weight: torch.Tensor) -> None:
            self.load_row_parallel_weight(loaded_weight)

    param_mod.SharedWeightParameter = FakeSharedWeightParameter

    class FakeLinearTransformMethod:
        pass

    class FakeHadamardTransform(torch.nn.Module):
        def __init__(
            self, transforms, layer, weight_loader, input_size_per_partition, output_partition_sizes
        ):
            super().__init__()
            if ht_mod.get_tensor_model_parallel_world_size() > 1:
                raise NotImplementedError("Online transforms with tensor parallelism is not supported")
            self.transforms = transforms
            head_dim = int(transforms[0].scheme.head_dim)
            self.weight = FakeSharedWeightParameter(head_dim=head_dim)

        def forward(self, value: torch.Tensor, part_id: int = 0) -> torch.Tensor:
            head_dim = int(self.transforms[part_id].scheme.head_dim)
            return fast_block_hadamard(value, head_dim)

    lin_mod.CompressedTensorsLinearTransformMethod = FakeLinearTransformMethod
    ht_mod.HadamardTransform = FakeHadamardTransform

    for name, mod in {
        "vllm.distributed.parallel_state": par_mod,
        "vllm.model_executor.parameter": param_mod,
        "vllm.distributed.communication_op": comm_mod,
        "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors": ct_mod,
        "vllm.model_executor.layers.quantization.compressed_tensors.transform.linear": lin_mod,
        "vllm.model_executor.layers.quantization.compressed_tensors.transform.module": ht_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    _patch_ct_linear_transform()

    tfm_16 = {
        0: types.SimpleNamespace(
            scheme=types.SimpleNamespace(type="hadamard", head_dim=16),
            args=types.SimpleNamespace(location=TransformLocation.INPUT),
        )
    }

    # Vision linear_fc2: K=4304 = 16*269 -> 2152 per rank under tp=2 (2152 % 16 == 8)
    layer_r0 = types.SimpleNamespace(tp_size=2, tp_rank=0)
    layer_r1 = types.SimpleNamespace(tp_size=2, tp_rank=1)
    ht_r0 = FakeHadamardTransform(tfm_16, layer_r0, None, 2152, [1152])
    ht_r1 = FakeHadamardTransform(tfm_16, layer_r1, None, 2152, [1152])

    # Simulate LinearBase.update_param_tp_status() overwriting partition.tp_rank = 1 on rank 1
    ht_r1.weight.partitions[0].tp_rank = 1
    ht_r1.weight.load_row_parallel_weight(torch.eye(16))
    assert ht_r1.weight.partitions[0].tp_rank == 0

    full_x = torch.randn(3, 4304, dtype=torch.float32)
    x_r0 = full_x[:, :2152].contiguous()
    x_r1 = full_x[:, 2152:].contiguous()
    boundary_full = torch.cat((x_r0[:, -8:], x_r1[:, :8]), dim=-1)
    comm_mod.tensor_model_parallel_all_gather = lambda t, dim=-1: boundary_full

    out_r0 = ht_r0(x_r0)
    out_r1 = ht_r1(x_r1)
    out_combined = torch.cat((out_r0, out_r1), dim=-1)
    ref = fast_block_hadamard(full_x, 16)
    torch.testing.assert_close(out_combined, ref, rtol=1e-5, atol=1e-5)

    # Also verify generic full all_gather fallback (e.g., when _warp_quant_partner_layer is None)
    ht_r0._warp_quant_partner_layer = None
    ht_r1._warp_quant_partner_layer = None
    comm_mod.tensor_model_parallel_all_gather = lambda t, dim=-1: full_x
    out_r0_fb = ht_r0(x_r0)
    out_r1_fb = ht_r1(x_r1)
    torch.testing.assert_close(torch.cat((out_r0_fb, out_r1_fb), dim=-1), ref, rtol=1e-5, atol=1e-5)


def test_qwen4_exp_ple_lut4_modern_vocab_parallel_embedding_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import types

    class FakePLEMethod:
        @staticmethod
        def from_quant_config(quant_config: object, *args: object, **kwargs: object) -> FakePLEMethod:
            del quant_config, args, kwargs
            return FakePLEMethod()

    class FakeVocabEmbedding(torch.nn.Module):
        def __init__(self, method: CustomLut4PLEMethod, num_rows: int = 32, dim: int = 160) -> None:
            super().__init__()
            self.embedding_method = method
            self.org_vocab_size = num_rows
            self.embedding_dim = dim
            self.params_dtype = torch.bfloat16
            self.shard_indices = types.SimpleNamespace(org_vocab_start_index=0, org_vocab_end_index=num_rows)
            method.create_weights(
                self,
                input_size_per_partition=dim,
                output_partition_sizes=[num_rows],
                input_size=dim,
                output_size=num_rows,
                params_dtype=torch.bfloat16,
                weight_loader=self.weight_loader,
            )

        def weight_loader(
            self, param: torch.Tensor, loaded_weight: torch.Tensor, checkpoint_start: int = 0
        ) -> None:
            rows = loaded_weight.shape[0]
            param.data[checkpoint_start : checkpoint_start + rows].copy_(loaded_weight)

    class FakeModernNGramEmbedding:
        def __init__(self, embedding: FakeVocabEmbedding) -> None:
            self.ngram_embedding = embedding
            self.split_ngram_parts = 2
            self.head_dim = 160

        def load_weights(self, weights: object) -> set[str]:
            return {name for name, _ in weights}  # type: ignore[misc]

    common_mod = types.ModuleType("vllm.models.qwen4_exp.common.ngram_embedding")
    common_mod.Qwen4ExpPLEEmbeddingMethod = FakePLEMethod  # type: ignore[attr-defined]
    nvidia_mod = types.ModuleType("vllm.models.qwen4_exp.nvidia.ngram_embedding")
    nvidia_mod.Qwen4ExpNGramEmbedding = FakeModernNGramEmbedding  # type: ignore[attr-defined]

    for name, mod in {
        "vllm": types.ModuleType("vllm"),
        "vllm.models": types.ModuleType("vllm.models"),
        "vllm.models.qwen4_exp": types.ModuleType("vllm.models.qwen4_exp"),
        "vllm.models.qwen4_exp.common": types.ModuleType("vllm.models.qwen4_exp.common"),
        "vllm.models.qwen4_exp.common.ngram_embedding": common_mod,
        "vllm.models.qwen4_exp.nvidia": types.ModuleType("vllm.models.qwen4_exp.nvidia"),
        "vllm.models.qwen4_exp.nvidia.ngram_embedding": nvidia_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    _patch_qwen4_exp_ple_lut4()

    qcfg = types.SimpleNamespace(ple_quantization={"format": "custom_lut4_g32_fp16scale", "group_size": 32})
    method = FakePLEMethod.from_quant_config(qcfg, "model.layers.0.ple.ngram_embedding", None)
    assert isinstance(method, CustomLut4PLEMethod)

    emb = FakeVocabEmbedding(method, num_rows=32, dim=160)
    ngram = FakeModernNGramEmbedding(emb)

    torch.manual_seed(7)
    q0 = torch.randint(0, 16, (16, 160), dtype=torch.uint8)
    s0 = torch.full((16, 5), 0.5, dtype=torch.float16)
    l0 = torch.linspace(-1.0, 1.0, 16, dtype=torch.float32)
    q1 = torch.randint(0, 16, (16, 160), dtype=torch.uint8)
    s1 = torch.full((16, 5), 0.25, dtype=torch.float16)
    l1 = torch.linspace(-2.0, 2.0, 16, dtype=torch.float32)
    p0 = (q0[:, 0::2] | (q0[:, 1::2] << 4)).contiguous()
    p1 = (q1[:, 0::2] | (q1[:, 1::2] << 4)).contiguous()

    loaded = ngram.load_weights(
        [
            ("layer_multipliers", torch.ones(3, dtype=torch.long)),
            ("ngram_embedding.shard_0.weight_packed", p0),
            ("ngram_embedding.shard_0.weight_scale", s0),
            ("ngram_embedding.shard_0.weight_lut", l0),
            ("ngram_embedding.shard_1.weight_packed", p1),
            ("ngram_embedding.shard_1.weight_scale", s1),
            ("ngram_embedding.shard_1.weight_lut", l1),
        ]
    )
    assert "ngram_embedding.weight" in loaded
    assert "layer_multipliers" in loaded

    ids = torch.tensor([[2, 19]], dtype=torch.long)
    out = method.embedding(emb, ids)
    assert out.shape == (1, 2, 160)
    ref0 = (l0[q0[2].long()].reshape(5, 32) * s0[2].unsqueeze(-1).float()).to(torch.bfloat16).reshape(160)
    ref1 = (l1[q1[3].long()].reshape(5, 32) * s1[3].unsqueeze(-1).float()).to(torch.bfloat16).reshape(160)
    torch.testing.assert_close(out[0, 0], ref0)
    torch.testing.assert_close(out[0, 1], ref1)

    pinned_out = torch.empty((2, 160), dtype=torch.bfloat16)
    method.lookup_from_pinned(emb, ids.reshape(-1), pinned_out)
    torch.testing.assert_close(pinned_out[0], ref0)
    torch.testing.assert_close(pinned_out[1], ref1)
    assert torch.equal(method.dequantize(emb, pinned_out, torch.bfloat16), pinned_out)


def test_stamp_vllm_autoload_normalizes_configs_and_synthesizes_preprocessors(
    tmp_path: pytest.TempPathFactory,
) -> None:
    import json
    from pathlib import Path

    from warp_quant.stamp import stamp_vllm_autoload

    d = Path(str(tmp_path))
    (d / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "text_config": {"model_type": "qwen4_exp_text"},
                "quantization_config": {
                    "ignore": ["re:^mtp\\.fc_.*"],
                    "ple_quantization": {"format": "lut4-gptq", "group_size": 32},
                },
            }
        ),
        encoding="utf-8",
    )
    (d / "processor_config.json").write_text(
        json.dumps(
            {
                "processor_class": "Qwen3VLProcessor",
                "image_processor": {"image_processor_type": "Qwen2VLImageProcessor"},
                "video_processor": {"video_processor_type": "Qwen3VLVideoProcessor"},
            }
        ),
        encoding="utf-8",
    )
    (d / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "Qwen3_5Tokenizer"}),
        encoding="utf-8",
    )

    stamp_vllm_autoload(d)

    assert (d / "vllm_warp_quant.py").is_file()
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    assert cfg["quantization_config"]["ignore"] == ["re:.*mtp\\.fc_.*"]
    assert cfg["text_config"]["ple_quantization"] == {"format": "lut4-gptq", "group_size": 32}
    assert "auto_map" not in cfg  # qwen4_exp uses processor/tokenizer auto_map

    pc = json.loads((d / "processor_config.json").read_text(encoding="utf-8"))
    assert pc["auto_map"]["AutoProcessor"] == "vllm_warp_quant.WarpQuantProcessor"
    assert (d / "preprocessor_config.json").is_file()
    assert (d / "video_preprocessor_config.json").is_file()

    tc = json.loads((d / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert tc["auto_map"]["AutoTokenizer"] == [
        "vllm_warp_quant.WarpQuantTokenizer",
        "vllm_warp_quant.WarpQuantTokenizer",
    ]


def test_w2_code_offset_scheme_propagation_and_stamp_vision_suffix(tmp_path) -> None:
    import json

    from warp_quant.stamp import ensure_processor_configs
    from warp_quant.vllm_plugin import _extract_w2_code_offset

    class _SchemeStub:
        _warp_quant_w2_code_offset = 0.5

    assert _extract_w2_code_offset(_SchemeStub()) == 0.5

    # transformers 5.19 emits model_type='qwen3_5_moe_vision' while base config has 'qwen3_5_moe'
    dst = tmp_path / "out"
    src = tmp_path / "base"
    dst.mkdir()
    src.mkdir()
    (dst / "config.json").write_text(
        json.dumps({"vision_config": {"model_type": "qwen3_5_moe_vision", "hidden_size": 1152}})
    )
    (src / "config.json").write_text(
        json.dumps({"vision_config": {"model_type": "qwen3_5_moe", "hidden_size": 1152}})
    )
    (src / "preprocessor_config.json").write_text(
        json.dumps({"image_processor_type": "Qwen2VLImageProcessor"})
    )
    added = ensure_processor_configs(dst, fallback=src)
    assert added == ["preprocessor_config.json"]
    assert (dst / "preprocessor_config.json").is_file()
