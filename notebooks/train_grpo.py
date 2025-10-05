import marimo

__generated_with = "0.16.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer
    return GRPOConfig, GRPOTrainer, load_dataset


@app.cell
def _(load_dataset):
    dataset = load_dataset("trl-lib/ultrafeedback-prompt", split="train")
    return (dataset,)


@app.cell
def _(GRPOConfig, GRPOTrainer, dataset):
    # Dummy reward function for demonstration purposes
    def reward_num_unique_letters(completions, **kwargs):
        """Reward function that rewards completions with more unique letters."""
        completion_contents = [completion[0]["content"] for completion in completions]
        return [float(len(set(content))) for content in completion_contents]

    training_args = GRPOConfig(
        output_dir="Qwen2-0.5B-GRPO",
        per_device_train_batch_size=32,
        )
    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_num_unique_letters,
        args=training_args,
        train_dataset=dataset,
    )
    return (trainer,)


@app.cell
def _(trainer):
    trainer.train()
    return


if __name__ == "__main__":
    app.run()
