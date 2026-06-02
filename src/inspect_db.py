# inspect_db.py

import chromadb

client = chromadb.PersistentClient(
    path="chroma_db"
)

collections = client.list_collections()

for c in collections:

    print("\nCollection:", c.name)

    data = c.get()

    print("Documents:",
          len(data["documents"]))

    print("\nSample Chunk:\n")
    print(data["documents"][0][:1000])