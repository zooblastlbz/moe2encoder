# Prompt Pair JSONL Schema

Each line is one JSON object with these fields:

- `anchor_text` (string, required)
- `positive_text` (string, required)
- `group_id` (string/int, optional)
- `prompt_type` (string, optional)

Example:

```json
{"anchor_text":"a red vintage car on a rainy street","positive_text":"a classic red car parked on a wet city road","group_id":"car_001","prompt_type":"object+style"}
```
