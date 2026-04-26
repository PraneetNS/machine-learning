def generate_answer(q):
    return "initial answer"

def critique(ans):
    return "this answer is vague, improve clarity"

def improve(ans, feedback):
    return "improved answer with clarity"

q = "Explain transformers"
ans = generate_answer(q)
fb = critique(ans)
final = improve(ans, fb)

print(final)