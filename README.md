<img width="736" height="708" alt="image" src="https://github.com/user-attachments/assets/c6dcdecb-c135-4427-8690-fbaf3b332d66" />


To download TinyStories from hugging face
```python
from datasets import load_dataset

dataset = load_dataset("roneneldan/TinyStories", split="train[:100000]")

with open("tinystories.txt", "w", encoding="utf-8") as f:
    for story in dataset["text"]:
        f.write(story)
        f.write("\n\n")

with open("tinystories.txt", "r", encoding="utf-8") as r:
    text = r.read()

print(text[:1000])
```
