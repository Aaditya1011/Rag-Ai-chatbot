from pathlib import Path
doc_id = '37ba6eb0-711e-4e00-b74-005c4645a808'
text_path = f"backend/data/extracted/{doc_id}.txt"

with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

print(text)