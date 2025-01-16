import copy

from itertools import product
from random import shuffle, seed

from aalpy.base.Oracle import Oracle
from aalpy.base.SUL import SUL


class WMethodEqOracleMealy(Oracle):
    """
    Equivalence oracle based on characterization set/ W-set. From 'Tsun S. Chow.   Testing software design modeled by
    finite-state machines'.
    """
    def __init__(self, alphabet: list, sul: SUL, extra_states, add_to_tree=False):
        """
        Args:

            alphabet: input alphabet
            sul: system under learning
            max_number_of_states: maximum number of states in the automaton
        """
        super().__init__(alphabet, sul)
        self.k = extra_states
        self.cache = set()
        self.add_to_tree = add_to_tree
        self.num_steps = 0
        self.resets = 0

    def find_cex(self, hypothesis, ob_tree=None, shuffle_seed=None):
        if not hypothesis.characterization_set:
            if len(hypothesis.states) == 1:
                hypothesis.characterization_set = [(a,) for a in self.alphabet]
            else:
                hypothesis.characterization_set = self.compute_characterization_set(hypothesis)

        shortest_paths = {state: hypothesis.get_shortest_path(hypothesis.initial_state, state) for state in hypothesis.states}
        transition_cover = [shortest_paths[state] + (letter,) for state in hypothesis.states for letter in self.alphabet]
        middle = (seq for i in range(self.k + 1) for seq in product(self.alphabet, repeat=i))

        test_suite = list(product(transition_cover, middle, hypothesis.characterization_set))

        if shuffle_seed is not None:
            seed(shuffle_seed)
        shuffle(test_suite)

        for seq in test_suite:
            inp_seq = tuple([i for sub in seq for i in sub])
            if inp_seq not in self.cache:
                self.reset_hyp_and_sul(hypothesis)
                self.resets += 1
                outputs = []
                for ind, letter in enumerate(inp_seq):
                    out_hyp = hypothesis.step(letter)
                    out_sul = self.sul.step(letter)
                    self.num_steps += 1
                    outputs.append(out_sul)
                    if out_hyp != out_sul:
                        self.sul.post()
                        return inp_seq[:ind + 1]
                
                if self.add_to_tree:
                    ob_tree.insert_observation(inp_seq, outputs)
                self.cache.add(inp_seq)

        return None

    def compute_characterization_set(self, hypothesis, char_set_init=None, online_suffix_closure=True, split_all_blocks=True, raise_warning=True):
        """
        Computation of a characterization set, that is, a set of sequences that can distinguish all states in the
        automation. This implementation is based on the aalpy implementation but is made compatible for a Mealy machien
        """
        blocks = list()
        blocks.append(copy.copy(hypothesis.states))
        char_set = [] if not char_set_init else char_set_init

        if char_set_init:
            for seq in char_set_init:
                blocks = hypothesis._split_blocks(blocks, seq)

        while True:
            try:
                block_to_split = next(b for b in blocks if len(b) > 1)
            except StopIteration:
                break

            split_state1, split_state2 = block_to_split[:2]
            dist_seq = hypothesis.find_distinguishing_seq(split_state1, split_state2, self.alphabet)

            if dist_seq is None:
                if raise_warning:
                    raise Exception("Automaton is non-canonical: could not compute characterization set."
                    "Returning None.")
                return None

            if online_suffix_closure:
                dist_seq_closure = [tuple(dist_seq[len(dist_seq) - i - 1:]) for i in range(len(dist_seq))]
            else:
                dist_seq_closure = [tuple(dist_seq)]

            if split_all_blocks:
                for seq in dist_seq_closure:
                    if seq in char_set:
                        continue
                    char_set.append(seq)
                    blocks = hypothesis._split_blocks(blocks, seq)
            else:
                blocks.remove(block_to_split)
                new_blocks = [block_to_split]
                for seq in dist_seq_closure:
                    char_set.append(seq)
                    new_blocks = self._split_blocks(new_blocks, seq)
                for new_block in new_blocks:
                    blocks.append(new_block)

        char_set = list(set(char_set))
        return char_set