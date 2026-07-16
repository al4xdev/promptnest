"""Smoke tests executed after installing the package with ``pip install -e``."""

from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from promptnest import LLMAdapter, PromptNest
from promptnest.adapters import CallableAdapter

ResultModel = TypeVar("ResultModel", bound=BaseModel)
FIXTURE = Path(__file__).parent / "fixtures" / "llm_responses.json"


class Summary(BaseModel):
    summary: str
    keywords: list[str]


class Report(BaseModel):
    full_summary: str
    all_keywords: list[str]


class InstalledPackageTest(unittest.TestCase):
    def test_public_api_with_known_json(self) -> None:
        responses = json.loads(FIXTURE.read_text(encoding="utf-8"))

        async def fake_llm(
            prompt: str,
            output_model: type[ResultModel],
            **kwargs: Any,
        ) -> str:
            self.assertIn("known", prompt)
            self.assertEqual(kwargs, {"model": "fixture"})
            key = "final" if output_model is Report else "chunk"
            return json.dumps(responses[key])

        adapter = CallableAdapter(fake_llm)
        self.assertIsInstance(adapter, LLMAdapter)

        async def execute() -> Report:
            runner = (
                PromptNest.have(adapter, {"document": ["known text"]})
                .set_llm_config(model="fixture")
                .set_retry_config(max_attempts=1, delay_s=0, timeout_s=1)
                .set_pre_prompt("known chunk: {chunk_text}", Summary)
                .set_pos_prompt("known partials: {partial_answers}", Report)
            )
            await runner.get_chunks_result()
            return await runner.run_pos_prompt()

        result = asyncio.run(execute())
        self.assertEqual(result, Report.model_validate(responses["final"]))


if __name__ == "__main__":
    unittest.main()
