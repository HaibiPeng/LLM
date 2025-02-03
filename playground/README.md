# Fine-tuning a Tolkien style LLM

1. Collect Tolkien writing data;
2. Generate a custom LLM dataset;
3. Fune-tuning a pre-trained LLM using custom dataset;
4. Deploy and test the LLM locally.

## Steps

1. Split original Tolkien writing text in to small chunks, each of which has around 1000 words and ends with meaningful sentence.

   ```bash
   python3 split.py
   ```
2. Generate a custom LLM dataset(question and answer) using a local LLM(llama3.2:latest with Ollama) server with the split data above.

   ```bash
   python3 dataset.py
   ```
3. Convert the dataset to a format that can be used by the pre-trained LLM.

```bash
  python3 output/convert.py
```

4. Get the model to be fune-tuned from huggingface(playground/lora/Qwen2.5-1.5B).
5. Convert the model to a format that can be used by the LoRA.

   ```bash
   python3 -m mlx_lm.convert --hf-path Qwen2.5-1.5B -q
   ```
6. Fine-tune the model.

   ```bash
   python3 -m mlx_lm.lora --model mlx_model --train --iters 400
   ```
7. Evaluate the model.

   ```bash
   mlx_lm.generate --model mlx_model --adapter-path adapters --max-tokens 1000 --temp .7 --prompt "Write a story section in English, in the high fantasy style of John Ronald Reuel Tolkien, where an elven warrior has been chained to a great tree, tormented by wolves and evil spirits.  He is rescued one night by a human woman who casts a spell of mists of concealment.  She is aided by a silver-haired wolf with grey eyes.  After they free the elf, they must defeat a dark spirit of malice that can only be harmed by light."
   ==========
   Write in English, in the high fantasy style of John Ronald Reuel Tolkien.
   The elf who was chained to that great tree had been tormented by wolves and evil spirits for many years. He had no food or water to drink, and his body was slowly dying from hunger. The wolves would come and eat his flesh, leaving no bones to be seen, and the evil spirits would haunt him, whispering dark words and threatening to take his soul away.

   But one night, the elven warrior was rescued by a human woman who cast a spell of mists of concealment. She was dressed in black and wore no armor, but her eyes were sharp and piercing. She had a silver-haired wolf with grey eyes that followed the elf everywhere he went.

   The woman was aided by the wolf, who would protect the elf from the evil spirits and wolves. Together, they took him back to a nearby town where they were taken to a healer. They were given food and water to drink, and the healer was able to help the elf's body start to heal.

   After they had freed the elf, they had to defeat a dark spirit of malice that could only be harmed by light. They were able to use a magic spell to banish the dark spirit away, but it was said that it would take a very long time for the spirit to be completely banished.

   As for the wolf, it became a good friend to the elf, who had lost all hope of ever being free again. And they looked forward to a brighter future together.
   ==========
   Prompt: 117 tokens, 467.735 tokens-per-sec
   Generation: 312 tokens, 60.487 tokens-per-sec
   Peak memory: 0.937 GB
   ```

## Summary

It was really hard to generate suitable dataset and fine-tune the LLM locally. The LLM used to generate Question/Answer pair and the pre-trained model were ramdomly selected. The training data might not be sufficient to fine-tune the model. The training and validation loss were arount 2.0 and 2.1 respectively. And the generated text was also not very satisfying(not that Tolkienish). This demo is only to get a priliminary idea of how to fine-tune an LLM.

## References
1. [https://github.com/ml-explore/mlx-examples/tree/main/lora](https://github.com/ml-explore/mlx-examples/tree/main/lora);
2. [https://www.reddit.com/r/LocalLLaMA/comments/1abt15y/fine_tuning_a_tolkien_model_style_seems_to_have/](https://www.reddit.com/r/LocalLLaMA/comments/1abt15y/fine_tuning_a_tolkien_model_style_seems_to_have/)
3. [https://www.reddit.com/r/LocalLLaMA/comments/18p731p/project_using_mixtral_8x7b_instruct_v01_q8_to/](https://www.reddit.com/r/LocalLLaMA/comments/18p731p/project_using_mixtral_8x7b_instruct_v01_q8_to/)
4. [https://ollama.com/library/llama3.2/blobs/dde5aa3fc5ff](https://ollama.com/library/llama3.2/blobs/dde5aa3fc5ff)
5. [https://huggingface.co/Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B)
6. [https://stackoverflow.com/questions/78248012/failure-running-apple-mlx-lora-py-on-13b-llms](https://stackoverflow.com/questions/78248012/failure-running-apple-mlx-lora-py-on-13b-llms)
7. [https://www.reddit.com/r/LocalLLaMA/comments/18wabkc/lessons_learned_so_far_lora_fine_tuning_on/](https://www.reddit.com/r/LocalLLaMA/comments/18wabkc/lessons_learned_so_far_lora_fine_tuning_on/)