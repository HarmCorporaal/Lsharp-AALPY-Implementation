from aalpy.base import Oracle, SUL
from ObservationTree import ObservationTree
from aalpy.automata import MealyMachine, MealyState
from WMethodEqOracleMealy import WMethodEqOracleMealy
from Apartness import Apartness
from ADS import Ads

class Lsharp:
    def __init__(self, alphabet: set, sul: SUL, eq_oracle: Oracle, extension_rule="Nothing", separation_rule="SepSeq", seed=None, 
               max_learning_rounds=None):
        """
        Args:
        alphabet: input alphabet
        sul: system under learning
        eq_oracle: equivalence oracle
        max_learning_rounds: number of learning rounds after which learning terminates.
        ob_tree: observation tree
        frontier_to_basis_dict: dictionary of the frontier states
        witness_cache: the witness cache
        apartness_cache: cache for the apartness computations
        extension_rule: Setting [Nothing, SepSeq, ADS]
        separation_rule: Setting [SepSeq, ADS]
        """
        self.alphabet = alphabet
        self.sul = sul
        self.eq_oracle = eq_oracle
        self.max_learning_rounds = max_learning_rounds
        self.ob_tree = ObservationTree(alphabet)
        self.ob_tree.root.reset_id_counter()
        self.basis = set()
        self.frontier_to_basis_dict = {}   
        self.basis_to_mealy_dict = {}
        self.witness_cache = {}
        self.extension_rule = extension_rule
        self.separation_rule = separation_rule
        self.results = [0,0,0,0,0]
        self.seed = seed
        
    def run_Lsharp(self):
        """
        Executes the L# algorithm (prefix-tree based automaton learning).
        Returns: Learned DFA or additional data if `return_data=True`.
        """
        assert self.extension_rule in {"Nothing", "SepSeq", "ADS"}
        assert self.separation_rule in {"SepSeq", "ADS"}

        learning_rounds = 0
        self.basis.add(self.ob_tree.root)

        self.sul.post()
        self.sul.pre()

        while True:
            if self.max_learning_rounds and learning_rounds == self.max_learning_rounds:
                break

            learning_rounds += 1

            hypothesis = self._build_hypothesis()

            # Added size for obtree
            self.results[4] = self.ob_tree.root._id_counter

            if (len(self.sul.automaton.states) == len(hypothesis.states)):
                return hypothesis, self.results, learning_rounds

            if isinstance(self.eq_oracle, WMethodEqOracleMealy):
                counter_example = self.eq_oracle.find_cex(hypothesis, self.ob_tree, self.seed)
                self.results[2] = self.eq_oracle.resets
                self.results[3] = self.eq_oracle.num_steps
            else:
                counter_example = self.eq_oracle.find_cex(hypothesis)
                if counter_example is not None:
                    self.results[2] = self.eq_oracle.num_queries
                    self.results[3] = self.eq_oracle.num_steps

            if counter_example is None:
                return hypothesis, self.results, learning_rounds

            cex_outputs = self.sul.query(counter_example)
            self._process_counter_example(hypothesis, counter_example, cex_outputs)

        raise Exception("Exceeded Max number of learning rounds")

    def _build_hypothesis(self):
        """
        Builds the hypothesis which will be sent to the SUL
        """
        build_hyp = 1
        while True:
            build_hyp += 1

            self._make_observation_tree_adequate()

            hypothesis = self._construct_hypothesis()

            counter_example = Apartness.compute_witness_in_tree_and_hypothesis(self.ob_tree, hypothesis)

            if not counter_example:
                return hypothesis

            cex_outputs = self.ob_tree.get_observation(counter_example)
            self._process_counter_example(hypothesis, counter_example, cex_outputs)


    def _make_observation_tree_adequate(self):
        """
        Updates the frontier and basis based on extension and separation rule
        """
        self._update_frontier_and_basis()
        while(not self._is_observation_tree_adequate()):
            self._make_basis_complete()
            self._make_frontiers_identified()
            self._promote_frontier_state()


    def _update_frontier_and_basis(self):
        """
        Updates the frontier to basis map, promotes a frontier state and checks for consistency
        """
        self._update_frontier_to_basis_dict()
        self._promote_frontier_state()
        self._check_frontier_consistency()
        self._update_frontier_to_basis_dict()

    
    def _update_basis_candidates(self, frontier_state):
        """
        Updates the basis candidates for the specified frontier state.
        Removes basis states that are deemed apart from the frontier state.
        """
        if frontier_state not in self.frontier_to_basis_dict:
            print(f"Warning: {frontier_state} not found in frontier_to_basis_dict.")
            return
        
        basis_list = self.frontier_to_basis_dict[frontier_state]
        self.frontier_to_basis_dict[frontier_state] = [
            basis_state for basis_state in basis_list 
            if not Apartness.states_are_apart(frontier_state, basis_state, self.ob_tree)
        ]

    def _update_frontier_to_basis_dict(self):
        """
        Checks for basis candidates (basis states with the same behavior) for each frontier state.
        If a frontier state and a basis state are "apart", the basis state is removed from the basis list.
        """
        for frontier_state, basis_list in self.frontier_to_basis_dict.items():
            self.frontier_to_basis_dict[frontier_state] = [
                basis_state for basis_state in basis_list 
                if not Apartness.states_are_apart(frontier_state, basis_state, self.ob_tree)
            ]               

    def _promote_frontier_state(self):
        """
        Searches for a isolated frontier state and adds it to the basis states if it is not associated with another basis state
        """
        for iso_frontier_state, basis_list in self.frontier_to_basis_dict.items():
            if not basis_list:
                new_basis = iso_frontier_state
                self.basis.add(new_basis)
                self.frontier_to_basis_dict.pop(new_basis)

                for frontier_state, new_basis_list in self.frontier_to_basis_dict.items():
                    if not Apartness.states_are_apart(new_basis, frontier_state, self.ob_tree):
                        new_basis_list.append(new_basis)
                break       

    def _check_frontier_consistency(self):
        """
        Checks if all the states are correctly defined and creates new frontier states when possible 
        """
        for basis_state in self.basis:
            for input in self.alphabet:
                maybe_frontier = basis_state.get_successor(input)
                if (maybe_frontier == None or maybe_frontier in self.basis or maybe_frontier in self.frontier_to_basis_dict):
                    continue
                
                self.frontier_to_basis_dict[maybe_frontier] = [
                    new_basis_state for new_basis_state in self.basis 
                    if not Apartness.states_are_apart(new_basis_state, maybe_frontier, self.ob_tree)
                ]

    def _is_observation_tree_adequate(self):
        """
        Check if the frontier state have only 1 basis candidate, and if all basis states have some output for every input.
        """
        self._check_frontier_consistency()
        for _, basis_list in self.frontier_to_basis_dict.items():
            if len(basis_list) != 1:
                return False
        
        for basis_state in self.basis:
            for input in self.alphabet:
                if basis_state.get_output(input) is None:
                    return False
        
        return True

    def _make_basis_complete(self):
        """
        Explore new frontier states and adding them to the frontier to basis map
        """
        for basis_state in self.basis:
            for input in self.alphabet:
                if basis_state.get_successor(input) is None:
                    self._explore_frontier(basis_state, input)
                    new_frontier = basis_state.get_successor(input)
                    basis_candidates = self._find_basis_candidates(new_frontier)
                    self.frontier_to_basis_dict[new_frontier] = basis_candidates

    def _find_basis_candidates(self, new_frontier):
        return (
            new_basis_state for new_basis_state in self.basis
            if not Apartness.states_are_apart(new_basis_state, new_frontier, self.ob_tree)
        )


    def _explore_frontier(self, basis_state, input):
        """
        explores a specific frontier state (basis state + input) by passing a query to the sul
        """
        if (self.extension_rule == "ADS"):
            suffix = Ads(self.ob_tree, self.basis)
            ads_in, ads_out = self._adaptive_output_query(self.ob_tree.get_transfer_sequence(self.ob_tree.root, basis_state), input, suffix)
            self.ob_tree.insert_observation(ads_in, ads_out)
            return 

        if (self.extension_rule == "Nothing" or (self.extension_rule == "SepSeq" and len(self.basis) == 1)):
            inputs = self.ob_tree.get_transfer_sequence(self.ob_tree.root, basis_state)
            inputs.append(input)
            outputs = self.sul.query(inputs) # LEARNING
            self.results[0] += 1
            self.results[1] += len(inputs)
            self.ob_tree.insert_observation(inputs, outputs)
            return

        if (self.extension_rule == "SepSeq"):
            iterator = iter(self.basis)
            basis_one = next(iterator)
            basis_two = next(iterator)

            witness = self._get_or_compute_witness(basis_one, basis_two)
            inputs = self.ob_tree.get_transfer_sequence(self.ob_tree.root, basis_state)
            inputs.append(input)
            inputs.extend(witness)
            outputs = self.sul.query(inputs) # LEARNING
            self.results[0] += 1
            self.results[1] += len(inputs)
            self.ob_tree.insert_observation(inputs, outputs)
            return

    def _adaptive_output_query(self, prefix, infix, suffix):
        """
        Adds input to the prefix and calls the base function
        """
        prefix.append(infix)
        return self._adaptive_output_query_base(prefix, suffix)

    def _adaptive_output_query_base(self, prefix, suffix):
        """
        Query the tree for a result, if unsuccesful query the sul and update the tree
        """
        from_node = self.ob_tree.get_successor(prefix)
        if from_node:
            tree_in, tree_out = self._answer_ads_from_tree(suffix, from_node)
            suffix.reset_to_root()

            if tree_out:
                inputs = prefix
                outputs = self.ob_tree.get_observation(prefix)
                inputs.extend(tree_in)
                outputs.extend(tree_out)
                return (inputs, outputs)

        # removed prefix call here
        sul_in, sul_out = self._sul_adaptive_query(prefix, suffix)
        if sul_out:
            outputs = sul_out
        
        self.ob_tree.insert_observation(sul_in, outputs)

        return sul_in, outputs
    
    def _sul_adaptive_query(self, inputs, ads):
        """
        sends inputs to the sul and returns the output
        """
        outputs_received = []
        last_output = None

        self.sul.post()
        self.sul.pre()

        for input in inputs:
            output = self.sul.step(input)
            # removed reset
            self.results[1] += 1
            outputs_received.append(output)

        while True:
            next_input = ads.next_input(last_output)
            if next_input is None:
                break
            inputs.append(next_input)
            output = self.sul.step(next_input)
            # removed reset
            self.results[1] += 1
            outputs_received.append(output)
            last_output = output
        self.results[0] += 1

        return inputs, outputs_received

    def _answer_ads_from_tree(self, ads, from_node):
        """
        searches the tree based on the inputs returning the inputs/ouputs when all ads inputs are used
        """
        prev_output = None
        inputs_sent = []
        outputs_received = []
        current_node = from_node

        while True:
            next_input = ads.next_input(prev_output)
            if next_input is None:
                break
            inputs_sent.append(next_input)

            output_from_node = current_node.get_output(next_input)
            successor_from_node = current_node.get_successor(next_input)
            if successor_from_node is None:
                return None, None

            prev_output = output_from_node
            outputs_received.append(output_from_node)
            current_node = successor_from_node

        ads.reset_to_root()
        return inputs_sent, outputs_received

    def _get_or_compute_witness(self, state_one, state_two):
        """
        Get witness by checking cache and computing it otherwise
        """
        pair_one = str(state_one.id) + "-" + str(state_two.id)
        if (pair_one in self.witness_cache):
            return self.witness_cache.get(pair_one)
        
        pair_two = str(state_two.id) + "-" + str(state_one.id)
        if (pair_two in self.witness_cache):
            return self.witness_cache.get(pair_two)
        
        witness = Apartness.compute_witness(state_one, state_two, self.ob_tree)
        self.witness_cache[pair_one] = witness
        self.witness_cache[pair_two] = witness
        return witness
    
    def _make_frontiers_identified(self):
        """
        Loop over all frontier states to indentify them
        """
        for frontier_state in self.frontier_to_basis_dict:
            self._identify_frontier(frontier_state)

    def _identify_frontier(self, frontier_state):
        """
        Identify a specific frontier state
        """
        if frontier_state not in self.frontier_to_basis_dict:
            raise Exception(f"Warning: {frontier_state} not found in frontier_to_basis_dict.")
        
        self._update_basis_candidates(frontier_state)
        old_candidate_size = len(self.frontier_to_basis_dict.get(frontier_state))
        if (old_candidate_size < 2):
            return
        
        if (self.separation_rule == "SepSeq" or old_candidate_size == 2):
            inputs, outputs = self._identify_frontier_sepseq(frontier_state)
        else:
            inputs, outputs = self._identify_frontier_ads(frontier_state)

        self.ob_tree.insert_observation(inputs, outputs)
        self._update_basis_candidates(frontier_state)
        if (len(self.frontier_to_basis_dict.get(frontier_state)) == old_candidate_size):
            print("specific identification did not increase the norm")

    def _identify_frontier_sepseq(self, frontier_state):
        """
        Specifically identify using sepseq
        """
        basis_candidates = self.frontier_to_basis_dict.get(frontier_state)
        iterator = basis_candidates
        basis_one = iterator[0]
        basis_two = iterator[1]

        witness = self._get_or_compute_witness(basis_one, basis_two)

        inputs = self.ob_tree.get_transfer_sequence(self.ob_tree.root, frontier_state)
        inputs.extend(witness)

        outputs = self.sul.query(inputs) # LEARNING
        self.results[0] += 1
        self.results[1] += len(inputs)

        return inputs, outputs

    def _identify_frontier_ads(self, frontier_state):
        """
        Specifically indentify using ADS
        """
        basis_candidates = self.frontier_to_basis_dict.get(frontier_state)
        suffix = Ads(self.ob_tree, set(basis_candidates))
        return self._adaptive_output_query_base(self.ob_tree.get_transfer_sequence(self.ob_tree.root, frontier_state), suffix)

    def _construct_hypothesis(self):
        """
        Construct a hypothesis (Mealy Machine) based on the observation tree
        """
        self.basis_to_mealy_dict.clear()
        state_counter = 0
        for basis_state in self.basis:
            state_id = f's{state_counter}'
            self.basis_to_mealy_dict[basis_state] = MealyState(state_id)
            state_counter += 1

        mealy_states = []
        for basis_state in self.basis:
            source = self.basis_to_mealy_dict[basis_state]
            for input_val in self.alphabet:
                output = basis_state.get_output(input_val)
                successor = basis_state.get_successor(input_val)

                if successor in self.frontier_to_basis_dict:
                    candidates = self.frontier_to_basis_dict[successor]
                    if len(candidates) > 1:
                        raise RuntimeError("Multiple basis candidates for a single frontier state.")
                    successor = next(iter(candidates))

                if successor not in self.basis_to_mealy_dict:
                    raise RuntimeError("Successor is not in the basisToStateMap.")
                
                source.output_fun[input_val] = output
                destination = self.basis_to_mealy_dict[successor]
                source.transitions[input_val] = destination
            
            mealy_states.append(source)

        hypothesis = MealyMachine(self.basis_to_mealy_dict[self.ob_tree.root], mealy_states)
        return hypothesis

    def _process_counter_example(self, hypothesis, cex_inputs, cex_outputs):
        """
        Inserts the counter example into the observation tree and searches for the input-output sequence which is different
        """
        self.ob_tree.insert_observation(cex_inputs, cex_outputs)
        hyp_outputs = hypothesis.compute_output_seq(hypothesis.initial_state, cex_inputs)
        prefix_index = self._get_counter_example_prefix_index(cex_outputs, hyp_outputs)
        self._process_binary_search(hypothesis, cex_inputs[:prefix_index], cex_outputs[:prefix_index])

    def _get_counter_example_prefix_index(self, cex_outputs, hyp_outputs):
        """
        Checks at which index the output functions differ
        """
        for index in range(len(cex_outputs)):
            if cex_outputs[index] != hyp_outputs[index]:
                return index
        raise RuntimeError("counter example and hypothesis outputs are equal")
    
    def _process_binary_search(self, hypothesis, cex_inputs, cex_outputs):
        """
        use binary search on the counter example to compute a witness between the real system and the hypothesis
        """
        tree_node = self.ob_tree.get_successor(cex_inputs)
        self._update_frontier_and_basis()

        if(tree_node in self.frontier_to_basis_dict or tree_node in self.basis):
            return
        
        hyp_state = self._get_mealy_successor(hypothesis, hypothesis.initial_state, cex_inputs)
        hyp_node = list(self.basis_to_mealy_dict.keys())[list(self.basis_to_mealy_dict.values()).index(hyp_state)]

        prefix = []
        current_state = self.ob_tree.root
        for input in cex_inputs:
            if current_state in self.frontier_to_basis_dict:
                break
            current_state = current_state.get_successor(input)
            prefix.append(input)

        h = (len(prefix) + len(cex_inputs)) // 2
        sigma1 = list(cex_inputs[:h])
        sigma2 = list(cex_inputs[h:])

        hyp_state_p = self._get_mealy_successor(hypothesis, hypothesis.initial_state, sigma1)
        hyp_node_p = list(self.basis_to_mealy_dict.keys())[list(self.basis_to_mealy_dict.values()).index(hyp_state_p)]
        hyp_p_access = self.ob_tree.get_transfer_sequence(self.ob_tree.root, hyp_node_p)

        witness = Apartness.compute_witness(tree_node, hyp_node, self.ob_tree)
        if witness is None:
            raise RuntimeError("Binary search: There should be a witness")

        query_inputs = hyp_p_access + sigma2 + witness
        query_outputs = self.sul.query(query_inputs) # LEARNING
        self.results[0] += 1
        self.results[1] += len(query_inputs)

        self.ob_tree.insert_observation(query_inputs, query_outputs)

        tree_node_p = self.ob_tree.get_successor(sigma1)

        witness_p = Apartness.compute_witness(tree_node_p, hyp_node_p, self.ob_tree)

        if witness_p is not None:
            self._process_binary_search(hypothesis, sigma1, cex_outputs[:h])
        else:
            new_inputs = list(hyp_p_access) + sigma2
            self._process_binary_search(hypothesis, new_inputs, query_outputs[:len(new_inputs)])

    
    def _get_mealy_successor(self, mealy_machine, from_state, inputs):
        mealy_machine.current_state = from_state
        for input in inputs:
            mealy_machine.current_state = mealy_machine.current_state.transitions[input]
        
        return mealy_machine.current_state
