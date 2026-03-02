from retriever import Retriever
from generator import Generator

r = Retriever()
g = Generator()

query = "What is Proximal Policy Optimization (PPO)?"

chunks = r.search(query, top_k=15, final_k=5)
answer = g.generate(query, chunks)

print(answer)