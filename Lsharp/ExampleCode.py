from Lsharp import Lsharp
from aalpy.utils import load_automaton_from_file
from aalpy.SULs import MealySUL
from aalpy.oracles import PerfectKnowledgeEqOracle, StatePrefixEqOracle
from WMethodEqOracleMealy import WMethodEqOracleMealy
import timeit

# Input Dot file:
dot_file = 'Lsharp/DotFiles/10_learnresult_MasterCard_fix.dot'

mealy_machine = load_automaton_from_file(dot_file, automaton_type='mealy')
input_al = mealy_machine.get_input_alphabet()

sul_mealy = MealySUL(mealy_machine)

results = None

def run_lsharp():
    global results

    perfect_oracle = PerfectKnowledgeEqOracle(input_al, sul_mealy, mealy_machine)
    w_method_oracle = WMethodEqOracleMealy(input_al, sul_mealy, 2, add_to_tree=False)
    state_prefix_oracle = StatePrefixEqOracle(input_al, sul_mealy, 50, 100)

    L_sharp = Lsharp(input_al, sul_mealy, w_method_oracle, extension_rule="Nothing", separation_rule="SepSeq", max_learning_rounds=50)
    learned_automaton, results, learning_rounds = L_sharp.run_Lsharp()
    print(learned_automaton)

execution_time_new = timeit.timeit(run_lsharp, number=1)

print(f"learn_queries: {results[0]} learn_steps: {results[1]} test_queries: {results[2]} test_steps: {results[3]}")
print(f"Average Execution time for 1 run(s): {execution_time_new / 1} seconds")
# L_sharp.run_Lsharp()

