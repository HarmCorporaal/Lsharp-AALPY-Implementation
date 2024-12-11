from Lsharp import Lsharp
from aalpy.utils import load_automaton_from_file
from aalpy.SULs import MealySUL
from aalpy.oracles import PerfectKnowledgeEqOracle, StatePrefixEqOracle
from WMethodEqOracleMealy import WMethodEqOracleMealy
import timeit
import csv, os

result_sul = 0
learned_automaton = None

fields = ["model", "number_of_states", "learning_rounds", "number_of_inputs", "learn_queries", "learn_steps", "test_resets", "test_steps", "extension_rule", "seperation_rule", "time"]
folder = "Experiment Results"
result_file = "Experiment1 Perfect.csv"
file_path = os.path.join(folder, result_file)


if not os.path.exists(file_path):
    with open(file_path,mode="w",newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

mealy_machine = None
learning_rounds = None
input_al = None
results = None

def benchmark(dot_file, extension_rule, separation_rule):
    global mealy_machine, learning_rounds, input_al, results
    dot_file = f'Lsharp/DotFiles/{dot_file}.dot'

    mealy_machine = load_automaton_from_file(dot_file, automaton_type='mealy')
    input_al = mealy_machine.get_input_alphabet()

    sul_mealy = MealySUL(mealy_machine)

    perfect_oracle = PerfectKnowledgeEqOracle(input_al, sul_mealy, mealy_machine)
    # w_method_oracle = WMethodEqOracleMealy(input_al, sul_mealy, 2, add_to_tree=False)
    # state_prefix_oracle = StatePrefixEqOracle(input_al, sul_mealy, 50, 100)

    L_sharp = Lsharp(input_al, sul_mealy, perfect_oracle, extension_rule=extension_rule, separation_rule=separation_rule, max_learning_rounds=75)
    learned_automaton, results, learning_rounds = L_sharp.run_Lsharp()

tests = [("Nothing", "SepSeq"), ("SepSeq", "SepSeq"), ("ADS", "SepSeq"), ("Nothing", "ADS"), ("SepSeq", "ADS"), ("ADS", "ADS")]
models = ["ASN_learnresult_SecureCode Aut_fix", "1_learnresult_MasterCard_fix", "LoesTarget", "Rabo_learnresult_SecureCode_Aut_fix", 
          "Rabo_learnresult_MAESTRO_fix", "ASN_learnresult_MAESTRO_fix", "4_learnresult_MAESTRO_fix", "10_learnresult_MasterCard_fix", 
          "Volksbank_learnresult_MAESTRO_fix", "learnresult_fix", "TCP_FreeBSD_Client", "TCP_Windows8_Client", "TCP_Linux_Client", 
          "DropBear", "OpenSSH", "TCP_Windows8_Server", "TCP_FreeBSD_Server", "TCP_Linux_Server", "BitVise"]
          
for dot_file in models:
    for (extension_rule, separation_rule) in tests:
        for index in range(100):
            execution_time_new = timeit.timeit(lambda: benchmark(dot_file, extension_rule, separation_rule), number=1)
            with open(file_path,mode="a",newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=fields)
                    writer.writerow({"model": dot_file, 
                                    "number_of_states": len(mealy_machine.states), 
                                    "learning_rounds": learning_rounds, 
                                    "number_of_inputs": len(input_al), 
                                    "learn_queries": results[0], 
                                    "learn_steps": results[1], 
                                    "test_resets": results[2], 
                                    "test_steps": results[3], 
                                    "extension_rule": extension_rule, 
                                    "seperation_rule": separation_rule, 
                                    "time": execution_time_new
                    })


