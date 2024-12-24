from Lsharp import Lsharp
from aalpy.utils import load_automaton_from_file
from aalpy.SULs import MealySUL
from WMethodEqOracleMealy import WMethodEqOracleMealy

# Input Dot file:
filename = "LoesTarget"
dot_file = f'Lsharp/DotFiles/{filename}.dot'

mealy_machine = load_automaton_from_file(dot_file, automaton_type='mealy')
input_al = mealy_machine.get_input_alphabet()

sul_mealy = MealySUL(mealy_machine)
w_method_oracle = WMethodEqOracleMealy(input_al, sul_mealy, 2, add_to_tree=False)

L_sharp = Lsharp(input_al, sul_mealy, w_method_oracle, extension_rule="Nothing", separation_rule="SepSeq", max_learning_rounds=50)

learned_automaton, results, learning_rounds = L_sharp.run_Lsharp()
print(learned_automaton)

print(f"number of learning rounds: {learning_rounds}")
print(f"learn_queries: {results[0]} learn_steps: {results[1]} test_resets: {results[2]} test_steps: {results[3]}")