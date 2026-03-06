"""
Test Diffusion Language Models (dLLM) on GSM8K benchmark.

This test verifies the correctness of dLLM inference support for:
- d3LLM/d3LLM_LLaDA (needs_full_prefill, FullAttnMultiBlock algorithm)
- d3LLM/d3LLM_Dream (needs_full_prefill, FullAttnMultiBlock algorithm)
- inclusionAI/LLaDA2.0-mini (LowConfidence algorithm)
- inclusionAI/LLaDA2.1-mini (LowConfidence algorithm)
"""

import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

# Register for CI with estimated time (seconds)
register_cuda_ci(est_time=600, suite="stage-b-test-large-2-gpu")


class TestD3LLMLLaDA(CustomTestCase):
    """Test d3LLM/d3LLM_LLaDA with FullAttnMultiBlock algorithm."""

    @classmethod
    def setUpClass(cls):
        cls.model = "d3LLM/d3LLM_LLaDA"
        cls.base_url = DEFAULT_URL_FOR_TEST

        other_args = [
            "--trust-remote-code",
            "--tp-size",
            "1",
            "--mem-fraction-static",
            "0.9",
            "--max-running-requests",
            "4",
            "--attention-backend",
            "flashinfer",
            "--dllm-algorithm",
            "FullAttnMultiBlock",
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=100,
            max_new_tokens=256,
            parallel=64,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        self.assertGreater(metrics["accuracy"], 0.50)

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (d3LLM_LLaDA)\n"
                f"accuracy={metrics['accuracy']:.3f}\n"
                f"throughput={metrics['output_throughput']:.2f} token/s\n"
            )


class TestD3LLMDream(CustomTestCase):
    """Test d3LLM/d3LLM_Dream with FullAttnMultiBlock algorithm."""

    @classmethod
    def setUpClass(cls):
        cls.model = "d3LLM/d3LLM_Dream"
        cls.base_url = DEFAULT_URL_FOR_TEST

        other_args = [
            "--trust-remote-code",
            "--tp-size",
            "1",
            "--mem-fraction-static",
            "0.9",
            "--max-running-requests",
            "4",
            "--attention-backend",
            "flashinfer",
            "--dllm-algorithm",
            "FullAttnMultiBlock",
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=100,
            max_new_tokens=256,
            parallel=64,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        self.assertGreater(metrics["accuracy"], 0.50)

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (d3LLM_Dream)\n"
                f"accuracy={metrics['accuracy']:.3f}\n"
                f"throughput={metrics['output_throughput']:.2f} token/s\n"
            )


class TestLLaDA20Mini(CustomTestCase):
    """Test inclusionAI/LLaDA2.0-mini with LowConfidence algorithm."""

    @classmethod
    def setUpClass(cls):
        cls.model = "inclusionAI/LLaDA2.0-mini"
        cls.base_url = DEFAULT_URL_FOR_TEST

        other_args = [
            "--trust-remote-code",
            "--tp-size",
            "1",
            "--mem-fraction-static",
            "0.9",
            "--max-running-requests",
            "4",
            "--attention-backend",
            "flashinfer",
            "--dllm-algorithm",
            "LowConfidence",
            "--cuda-graph-bs",
            "1",
            "2",
            "3",
            "4",
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=100,
            max_new_tokens=512,
            parallel=64,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        self.assertGreater(metrics["accuracy"], 0.85)

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (LLaDA2.0-mini)\n"
                f"accuracy={metrics['accuracy']:.3f}\n"
                f"throughput={metrics['output_throughput']:.2f} token/s\n"
            )


class TestLLaDA21Mini(CustomTestCase):
    """Test inclusionAI/LLaDA2.1-mini with LowConfidence algorithm."""

    @classmethod
    def setUpClass(cls):
        cls.model = "inclusionAI/LLaDA2.1-mini"
        cls.base_url = DEFAULT_URL_FOR_TEST

        other_args = [
            "--trust-remote-code",
            "--tp-size",
            "1",
            "--mem-fraction-static",
            "0.9",
            "--max-running-requests",
            "4",
            "--attention-backend",
            "flashinfer",
            "--dllm-algorithm",
            "LowConfidence",
            "--cuda-graph-bs",
            "1",
            "2",
            "3",
            "4",
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=100,
            max_new_tokens=512,
            parallel=64,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        self.assertGreater(metrics["accuracy"], 0.85)

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (LLaDA2.1-mini)\n"
                f"accuracy={metrics['accuracy']:.3f}\n"
                f"throughput={metrics['output_throughput']:.2f} token/s\n"
            )


if __name__ == "__main__":
    unittest.main()
