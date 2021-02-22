from fairseq.data import Dictionary


class DictionaryDiscreet(Dictionary):
    def add_symbol(self, word, n=1, verbose=False):
        """Adds a word to the dictionary"""
        if word is None:
            return None
        if word in self.indices:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            if verbose:
                print(f'| add new symbol "{word}"')
            return idx

    def pad_length_to_multiples(self, padding_factor=8, verbose=False):
        # there may already be madeupword in the dictionary
        threshold_nwords = len(self)
        i = len([symbol for symbol in self.symbols if symbol.startswith("madeupword")])
        while threshold_nwords % padding_factor != 0:
            symbol = "madeupword{:04d}".format(i)
            self.add_symbol(symbol, 0, verbose)
            i += 1
            threshold_nwords += 1
