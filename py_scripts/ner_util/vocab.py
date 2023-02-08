from collections import Counter

PAD = '___PAD___'
UNKNOWN = '___UNKNOWN___'
BOS = '___BOS___'
EOS = '___EOS___'

class Vocabulary:
    """Manages the numerical encoding of the vocabulary."""
    
    def __init__(self, max_voc_size=None, include_unknown=True, lower=False,
                 character=False):

        self.include_unknown = include_unknown
        self.dummies = [PAD, UNKNOWN, BOS, EOS] if self.include_unknown else [PAD, BOS, EOS]
        
        self.character = character
        
        # String-to-integer mapping
        self.stoi = None

        # Integer-to-string mapping
        self.itos = None

        # Maximally allowed vocabulary size.
        self.max_voc_size = max_voc_size

        self.lower = lower
        self.vectors = None
        
    def build(self, seqs):
        """Builds the vocabulary."""
        
        if self.character:
            seqs = [ [c for w in seq for c in w] for seq in seqs ]
        
        if self.lower:
            seqs = [ [s.lower() for s in seq] for seq in seqs ]
        
        # Sort all words by frequency
        word_freqs = Counter(w for seq in seqs for w in seq)
        word_freqs = sorted(((f, w) for w, f in word_freqs.items()), reverse=True)

        # Build the integer-to-string mapping. The vocabulary starts with the two dummy symbols,
        # and then all words, sorted by frequency. Optionally, limit the vocabulary size.
        
        if self.max_voc_size:
            self.itos = self.dummies + [ w for _, w in word_freqs[:self.max_voc_size-len(self.dummies)] ]
        else:
            self.itos = self.dummies + [ w for _, w in word_freqs ]

        # Build the string-to-integer map by just inverting the aforementioned map.
        self.stoi = { w: i for i, w in enumerate(self.itos) }

       
    def encode(self, seqs):
        """Encodes a set of documents."""
        unk = self.stoi.get(UNKNOWN)
        bos = self.stoi.get(BOS)
        eos = self.stoi.get(EOS)
        
        if self.character:
            if self.lower:
                seqs = [ [[c for c in w.lower()] for w in seq] for seq in seqs ]
            return [[[bos,eos]]+[[bos]+[self.stoi.get(c, unk) for c in w]+[eos] for w in seq]+[[bos,eos]] for seq in seqs]
        else:
            if self.lower:
                seqs = [ [s.lower() for s in seq] for seq in seqs ]
            return [[bos]+[self.stoi.get(w, unk) for w in seq]+[eos] for seq in seqs]

    def get_unknown_idx(self):
        """Returns the integer index of the special dummy word representing unknown words."""
        return self.stoi[UNKNOWN]
    
    def get_pad_idx(self):
        """Returns the integer index of the special padding dummy word."""
        return self.stoi[PAD]
    
    def __len__(self):
        return len(self.itos)