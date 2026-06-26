from core.network import Network

# VOCAB
words = [
    "how","are","you","hi","hello","ai","i","am","they","we",
    "what","is","your","name","my","good","bad","fine","thanks",
    "who","esc",
]
numbers = list(range(0,len(words)))
word2id = {a:b for a,b in zip(words,numbers)}

id2word = {v: k for k, v in word2id.items()}
vocab_size = len(word2id)

def encode(text):
    return [word2id[w] for w in text.split() if w in word2id]

# TRAINING DATA
data = [
    # how are you
    ([0], 1),
    ([0,1], 2),
    ([0,1,2], 20),
    # hi -> how are you
    ([3], 0),
    ([3,0], 1),
    ([3,0,1], 2),
    ([3,0,1,2], 20),

    # hello -> hi
    ([4], 3),
    ([4,3], 20),

    # i am ai
    ([6], 7),
    ([6,7], 5),
    ([6,7,5], 20),

    # they are good
    ([8], 1),
    ([8,1], 15),
    ([8,1,15], 20),

    # we are fine
    ([9], 1),
    ([9,1], 17),
    ([9,1,17], 20),

    # what is your name
    ([10], 11),
    ([10,11], 12),
    ([10,11,12], 13),
    ([10,11,12,13], 20),

    # my name ai
    ([14], 13),
    ([14,13], 5),
    ([14,13,5], 20),

    # how are you -> fine
    ([0,1,2], 17),
    ([0,1,2,17], 20),

    # how are you -> good
    ([0,1,2], 15),
    ([0,1,2,15], 20),

    # how are you -> bad
    ([0,1,2], 16),
    ([0,1,2,16], 20),

    # thanks
    ([18], 20),

    # who are you
    ([19], 1),
    ([19,1], 2),
    ([19,1,2], 20),
]


# MODEL
def createModel(vocab_size):
    net = Network(vocab_size=vocab_size,
    dim=64,lr=0.005,
    ff_hidden=128,)

    print("=== TRAINING ===")

    loss = net.train(data,
    epochs=2500,
    step=50,
    )

    net.save("model.qai")
    return net
    
def startModel(net):
    print("===Run===")

    while True:
        text = input("-> ").strip().lower()
        if text == "exit":
            break
        seq = encode(text)
        if not seq:
            print("Unknown input")
            continue
        generated = text
        for _ in range(20):
            pred = net.predict(seq)
            word = id2word[pred]
            if word == "esc":
                break
            print("Bot:", word)
            generated += " " + word
            seq = encode(generated)

if __name__ == "__main__":
    net = createModel(vocab_size)

    