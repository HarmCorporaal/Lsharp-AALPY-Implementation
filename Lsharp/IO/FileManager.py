import os
import pydot
from aalpy.automata import MealyMachine, MealyState

class FileManager:
    def __init__(self):
        pass

    # ---------------------------------------------
    # Load A Mealy Machine from a specific Dot file
    # Input: dotfile name
    # Output: Meale Machine
    # ---------------------------------------------
    def loadMealyFromDotFile(self, dotFile):
        cur_path = os.path.dirname(__file__)
        new_path = os.path.join(cur_path, f'../DotFiles/{dotFile}')
        print(new_path)
        graphs = pydot.graph_from_dot_file(new_path)
        graph = graphs[0] 

        states = {}
        initial_state_id = None

        # Identify initial state
        for edge in graph.get_edges():
            src = edge.get_source().strip('"')
            dst = edge.get_destination().strip('"')

            if src.startswith("__start"):
                initial_state_id = dst  
                break

        # Create all states
        for node in graph.get_nodes():
            state_id = node.get_name().strip('"')

            if state_id.startswith("__start"):
                continue

            if state_id not in states:
                states[state_id] = MealyState(state_id)

        # Define Transition / Outputs
        for edge in graph.get_edges():
            src = edge.get_source().strip('"')
            dst = edge.get_destination().strip('"')

            if src.startswith("__start"):
                continue

            label = edge.get_label().strip('"')

            if '/' in label:
                input_signal, output_signal = label.split('/')
            else:
                input_signal = label
                output_signal = ''

            if src not in states:
                states[src] = MealyState(src)
            if dst not in states:
                states[dst] = MealyState(dst)

            states[src].transitions[input_signal] = states[dst]     # Transition function δ
            states[src].output_fun[input_signal] = output_signal    # Output function λ

        # Create Mealy Machine
        mealy_states = list(states.values())
        initial_state = states[initial_state_id] if initial_state_id else mealy_states[0]

        mealy_machine = MealyMachine(initial_state, mealy_states)

        # Compute shortest path prefixes
        for state in mealy_states:
            state.prefix = mealy_machine.get_shortest_path(mealy_machine.initial_state, state)

        return mealy_machine


