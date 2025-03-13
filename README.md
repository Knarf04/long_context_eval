# Long Context Evaluation
## Phone Book Retrieval

The dataset is generated using `phone_book/phone_book_dataset.py` based on the following rules:

- The prompt templates are sourced from [this paper](https://arxiv.org/abs/2406.07887) (Section 3.3.3).
- Setting `reversed=True` places the few-shot examples and the query at the beginning of the generated text; otherwise, they appear at the end.
- Few-shot examples are evenly distributed throughout the generated phone book.
- The depth of the query is determined by the index (`idx`) of the item.
- If a tokenizer is provided, the generated sequence will have a total token count that is as close as possible to the specified length. Without a tokenizer, the length is interpreted as the number of entries.
- The script also generates **tokenized sequences** and **attention masks**.

To mitigate potential [OOM issues caused by FSDP](https://github.com/huggingface/transformers/issues/30228#issuecomment-2350022762), we utilize **FSDP2** for improved memory efficiency and usability.

Below is an example command to run the evaluation:

First, generate and store a dataset for a specific length
```bash
python phone_book_dataset.py \
    --model ibm-fms/Bamba-9B \
    --length 4096 \
    --size 400 \
    --few-shot 3 \
    --reversed \
    --save-path $HF_HOME/datasets/phonebook
```
Then, run evaluation
```bash
torchrun --nnodes 1 --nproc_per_node 4 \
    phone_book_eval.py \
    --model ibm-fms/Bamba-9B \
    --length-list 4096 \
    --batch-size 8 \
    --size 400 \
    --few-shot 3 \
    --reversed \
    --save-path $HF_HOME/datasets/phonebook
```
