import argparse
import time
from pathlib import Path

from openai import OpenAI


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm vLLM prefix cache")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="vLLM OpenAI-compatible endpoint",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument(
        "--system-prompt-file",
        default=None,
        help="path to the system prompt text used by the calls",
    )
    parser.add_argument("--user", default="Namaste", help="dummy user message")
    parser.add_argument("--rounds", type=int, default=3, help="repeat count to pin cache")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=300)
    if args.system_prompt_file:
        system = Path(args.system_prompt_file).read_text()
    else:
        system = (
            "You are Vaani, a warm, helpful Hindi voice assistant. "
            "Always open replies with a filler word such as 'Hmm' or 'Ji'."
        )
    for i in range(args.rounds):
        t0 = time.perf_counter()
        client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": args.user},
            ],
            max_tokens=1,
        )
        dt = (time.perf_counter() - t0) * 1000
        print(f"round {i + 1}: {dt:.1f} ms")


if __name__ == "__main__":
    main()