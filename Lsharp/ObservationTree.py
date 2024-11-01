class Node:
    _id_counter = 0

    def __init__(self, parent=None):
        Node._id_counter += 1
        self.id = Node._id_counter
        self.successors = {}
        self.parent = parent
        self.input_to_parent = None

    def __hash__(self):
        return hash(self.id)

    def add_successor(self, input_val, output_val, successor_node):
        """ Adds a successor node to the current node based on input """
        self.successors[input_val] = (output_val, successor_node)

    def get_successor(self, input_val):
        """ Returns the successor node for the given input """
        if input_val in self.successors:
            return self.successors[input_val][1]
        return None

    def get_output(self, input_val):
        """ Returns the output for the given input """
        if input_val in self.successors:
            return self.successors[input_val][0]
        return None
    
    def extend_and_get(self, input, output):
        """ Extend the node with a new successor and return the successor node """
        if (input in self.successors):
            out = self.successors[input][0]
            if out != output:
                raise Exception(f"observation not consistent with tree with output from tree: {out} and output from call: {output}")
            return self.successors[input][1]
        successor_node = Node(parent=self)
        self.add_successor(input, output, successor_node)
        successor_node.input_to_parent = input
        return successor_node


class ObservationTree:
    def __init__(self, alphabet):
        """
        Initialize the tree with a root node and the alphabet
        """
        self.root = Node()
        self.alphabet = set(alphabet)

    def _validate_input(self, inputs):
        """
        Check if all inputs are valid (part of the alphabet)
        """
        for input_val in inputs:
            if input_val not in self.alphabet:
                raise ValueError(f"Input '{input_val}' is not in the alphabet.")

    def insert_observation(self, inputs, outputs):
        """
        Insert an observation into the tree using sequences of inputs and outputs
        """
        if len(inputs) != len(outputs):
            raise ValueError("Inputs and outputs must have the same length.")
        
        self._validate_input(inputs)

        current_node = self.root
        for input_val, output_val in zip(inputs, outputs):
            current_node = current_node.extend_and_get(input_val, output_val)

    def get_observation(self, inputs):
        """
        Retrieve the list of outputs based on a given sequence of inputs
        """
        self._validate_input(inputs)

        current_node = self.root
        observation = []
        for input_val in inputs:
            output = current_node.get_output(input_val)
            if output is None:
                return None
            observation.append(output)
            current_node = current_node.get_successor(input_val)

        return observation

    def get_successor(self, inputs):
        """
        Retrieve the node (sub-tree) corresponding to the given sequence of inputs
        """
        self._validate_input(inputs)

        current_node = self.root
        for input_val in inputs:
            successor_node = current_node.get_successor(input_val)
            if successor_node is None:
                return None
            current_node = successor_node

        return current_node

    def get_transfer_sequence(self, from_node, to_node):
        """
        Get the transfer sequence (inputs) that moves from one node to another
        """
        transfer_sequence = []
        current_node = to_node

        while current_node != from_node:
            if current_node.parent is None:
                return None
            transfer_sequence.append(current_node.input_to_parent)
            current_node = current_node.parent

        transfer_sequence.reverse()
        return transfer_sequence