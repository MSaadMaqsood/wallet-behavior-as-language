
import math
from collections import Counter

BOS, EOS = "<BOS>", "<EOS>"


class KneserNeyLM:
    def __init__(self, n, discount=None, unigram_discount=True):
        self.n = n
        self.fixed_discount = discount            # None = estimate per order
        self.unigram_discount = unigram_discount  # False reproduces NLTK's unigrams

    # ---------------------------------------------------------------- training
    def fit(self, sentences):
        self.vocab = sorted({t for s in sentences for t in s} | {EOS})  # predictable tokens

        top = Counter()  # raw counts of the highest order
        for s in sentences:
            seq = self._pad(s)
            for i in range(self.n - 1, len(seq)):
                top[tuple(seq[i - self.n + 1:i + 1])] += 1

        self.counts = {self.n: top}
        for k in range(self.n - 1, 0, -1):  # continuation counts from the order above
            cont = Counter()
            for gram in self.counts[k + 1]:
                cont[gram[1:]] += 1
            self.counts[k] = cont

        self.context_total, self.context_types, self.discount = {}, {}, {}
        for k, table in self.counts.items():
            total, types = Counter(), Counter()
            for gram, c in table.items():
                total[gram[:-1]] += c
                types[gram[:-1]] += 1
            self.context_total[k], self.context_types[k] = total, types
            self.discount[k] = (self._estimate_discount(table) if self.fixed_discount is None
                                else self.fixed_discount)
        if not self.unigram_discount:
            self.discount[1] = 0.0
        return self

    @staticmethod
    def _estimate_discount(table):
        freq = Counter(table.values())
        n1, n2 = freq[1], freq[2]
        if n1 == 0 or n2 == 0:
            return 0.75  # too few rare n-grams to estimate (e.g. unigrams of a tiny vocabulary)
        return min(max(n1 / (n1 + 2 * n2), 0.05), 0.95)

    def _pad(self, sentence):
        return [BOS] * (self.n - 1) + list(sentence) + [EOS]

    # ----------------------------------------------------------------- scoring
    def prob(self, word, context):
        """P(word | context), where context = the preceding (padded) tokens."""
        p = 1.0 / len(self.vocab)  # order 0: uniform
        for k in range(1, self.n + 1):
            h = tuple(context[len(context) - k + 1:])  # the last k-1 tokens
            total = self.context_total[k].get(h, 0)
            if total == 0:
                continue  # unseen context: keep the shorter-context estimate
            d = self.discount[k]
            c = self.counts[k].get(h + (word,), 0)
            p = max(c - d, 0.0) / total + d * self.context_types[k][h] / total * p
        return p

    def token_logprobs(self, sentence):
        """Natural-log probability of every predicted token (actions + <EOS>)."""
        seq = self._pad(sentence)
        return [math.log(self.prob(seq[i], seq[i - self.n + 1:i]))
                for i in range(self.n - 1, len(seq))]

    def window_logprobs(self, tokens):
        """Log-probabilities of a window cut from the MIDDLE of a history: no
        <BOS>/<EOS>, each token conditioned only on the tokens before it inside
        the window (shorter contexts fall back to lower orders automatically)."""
        return [math.log(self.prob(tokens[i], tokens[:i])) for i in range(len(tokens))]

    def sentence_perplexity(self, sentence):
        lp = self.token_logprobs(sentence)
        return math.exp(-sum(lp) / len(lp))

    def perplexity(self, sentences):
        """Corpus perplexity: exp of the mean negative log-probability per token."""
        total, count = 0.0, 0
        for s in sentences:
            lp = self.token_logprobs(s)
            total += sum(lp)
            count += len(lp)
        return math.exp(-total / count)
