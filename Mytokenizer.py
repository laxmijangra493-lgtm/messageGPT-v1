from collections import Counter, defaultdict

def train_bpe(text, num_merges=1800):
    words = text.split()
    words_count = Counter(words)
    vocab = {tuple(list(word) + ['</w>']): count for word, count in words_count.items()}

    eachWord = set()
    for wt in vocab.keys():
        eachWord.update(wt)
    vocab_list = sorted(list(eachWord))
    merges = {}

    for i in range(num_merges):
        pairCount = Counter()
        where_pair = defaultdict(set)
        for w, freq in vocab.items():
            for j in range(len(w) - 1):
                pair = (w[j], w[j + 1])
                pairCount[pair] += freq
                where_pair[pair].add(w)
        if not pairCount:
            break

        best_pair, freq = pairCount.most_common(1)[0]
        merged_token = "".join(best_pair)
        merges[best_pair] = merged_token
        vocab_list.append(merged_token)

        new_vocab = {}
        wtu = where_pair[best_pair]
        for word_tuple, f in vocab.items():
            if word_tuple not in wtu:
                new_vocab[word_tuple] = f
                continue

            new_word = []
            j = 0
            while j < len(word_tuple):
                if j < len(word_tuple) - 1 and (word_tuple[j], word_tuple[j + 1]) == best_pair:
                    new_word.append(merged_token)
                    j += 2
                else:
                    new_word.append(word_tuple[j])
                    j += 1
            new_vocab[tuple(new_word)] = f
        vocab = new_vocab

    if "<unk>" not in vocab_list:
        vocab_list.append("<unk>")

    stoi = {s: i for i, s in enumerate(vocab_list)}
    itos = {i: s for i, s in enumerate(vocab_list)}

    return merges, stoi, itos