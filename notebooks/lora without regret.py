import marimo

__generated_with = "0.16.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # GRPO + LoRA Without Regret

    Train a language model using GRPO with LoRA on mathematical reasoning tasks.

    Based on LoRA Without Regret - use LoRA rank = 1, apply to all layers, learning rate ~1e-6.
    """
    )
    return


@app.cell
def _():
    import os
    from typing import Optional
    import torch
    from datasets import load_dataset
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify
    from trl import (
        GRPOConfig,
        GRPOTrainer,
        ModelConfig,
        get_peft_config,
        get_quantization_config,
        get_kbit_device_map,
    )

    os.environ["TRACKIO_SPACE_ID"] = "trl-lora-without-regret"
    os.environ["TRACKIO_PROJECT"] = "trl-lora-without-regret"

    return (
        GRPOConfig,
        GRPOTrainer,
        LatexExtractionConfig,
        ModelConfig,
        NormalizationConfig,
        Optional,
        get_kbit_device_map,
        get_peft_config,
        get_quantization_config,
        load_dataset,
        parse,
        torch,
        verify,
    )


@app.cell
def _(mo):
    mo.md(r"""# Configuration""")
    return


@app.cell
def _(GRPOConfig, ModelConfig):
    model_config = ModelConfig(
        model_name_or_path="Qwen/Qwen2-0.5B",
        torch_dtype="bfloat16",
        use_peft=True,
        lora_r=1,
        lora_alpha=32,
        lora_target_modules="all-linear",
        # load_in_4bit=True,
    )

    training_args = GRPOConfig(
        output_dir="./grpo-lora-qwen3",
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        max_steps=100,
        gradient_checkpointing=True,
        num_generations=8,
        generation_batch_size=16,  # Set generation_batch_size to be divisible by num_generations
        max_prompt_length=2048,
        max_completion_length=1024,
        logging_steps=10,
        save_steps=50,
        # report_to=["trackio"],
        bf16=True,
    )
    return model_config, training_args


@app.cell
def _(mo):
    mo.md(r"""# Dataset""")
    return


@app.cell
def _(load_dataset):
    dataset = load_dataset("HuggingFaceH4/OpenR1-Math-220k-default-verified", split="train")
    dataset = dataset.select(range(min(5000, len(dataset))))
    def make_conversation(example):
        prompt = [{"role": "user", "content": example["problem"]}]
        example["chat_template_kwargs"] = {"enable_thinking": False}
        return {"prompt": prompt}


    dataset = dataset.map(make_conversation)
    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col not in ["prompt", "solution"]]
    )
    return (dataset,)


@app.cell
def _(mo):
    mo.md(r"""# Reward Function""")
    return


@app.cell
def _(LatexExtractionConfig, NormalizationConfig, Optional, parse, verify):
    def strip_reasoning_accuracy_reward(
        completions: list[list[dict[str, str]]], solution: list[str], **kwargs
    ) -> list[Optional[float]]:
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            while "<think>" in content and "</think>" in content:
                start = content.find("<think>")
                end = content.find("</think>", start)
                if start != -1 and end != -1:
                    content = content[:start] + content[end + len("</think>") :]
                else:
                    break

            gold_parsed = parse(
                f"${sol}$",
                extraction_config=[
                    LatexExtractionConfig(
                        boxed_match_priority=0, try_extract_without_anchor=True
                    )
                ],
            )

            if len(gold_parsed) != 0:
                # We require the answer to be provided in correct latex (no malformed operators)
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            boxed_match_priority=0,
                            normalization_config=NormalizationConfig(
                                basic_latex=True,
                                units=True,
                                malformed_operators=False,
                                nits=False,
                                boxed=True,
                            ),
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )
                try:
                    reward = float(verify(gold_parsed, answer_parsed))
                except:
                    reward = None
            else:
                reward = None

            rewards.append(reward)
        return rewards

    return (strip_reasoning_accuracy_reward,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Initialize Trainer""")
    return


@app.cell
def _(
    GRPOTrainer,
    dataset,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    model_config,
    strip_reasoning_accuracy_reward,
    torch,
    training_args,
):
    dtype = (
        getattr(torch, model_config.torch_dtype)
        if model_config.torch_dtype not in ["auto", None]
        else model_config.torch_dtype
    )
    training_args.model_init_kwargs = {
        "torch_dtype": dtype,
        "device_map": get_kbit_device_map(),
        "quantization_config": get_quantization_config(model_config),
    }

    peft_config = get_peft_config(model_config)

    trainer = GRPOTrainer(
        model=model_config.model_name_or_path,
        args=training_args,
        reward_funcs=[strip_reasoning_accuracy_reward],
        train_dataset=dataset,
        peft_config=peft_config,
    )

    return (trainer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Train""")
    return


@app.cell
def _(trainer):
    trainer.train()
    return


if __name__ == "__main__":
    app.run()
