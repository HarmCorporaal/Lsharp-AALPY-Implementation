from Lsharp import Lsharp
from aalpy.utils import load_automaton_from_file
from aalpy.SULs import MealySUL
from aalpy.oracles import PerfectKnowledgeEqOracle, StatePrefixEqOracle
from WMethodEqOracleMealy import WMethodEqOracleMealy
import timeit
import csv, os

result_sul = 0
learned_automaton = None

fields = ["model", "number_of_states", "number_of_inputs", "complexity", "learning_rounds", "learn_resets", "learn_steps", "test_resets", "test_steps", "extension_rule", "separation_rule", "time", "seed"]
folder = "Experiment Results"
result_file = "Experiment2 - W-Method with buffer.csv"
file_path = os.path.join(folder, result_file)


if not os.path.exists(file_path):
    with open(file_path,mode="w",newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

mealy_machine = None
learning_rounds = None
input_al = None
results = None

def benchmark(dot_file, extension_rule, separation_rule, seed):
    global mealy_machine, learning_rounds, input_al, results
    dot_file = f'Lsharp/DotFiles/{dot_file}.dot'

    mealy_machine = load_automaton_from_file(dot_file, automaton_type='mealy')
    input_al = mealy_machine.get_input_alphabet()

    sul_mealy = MealySUL(mealy_machine)

    # perfect_oracle = PerfectKnowledgeEqOracle(input_al, sul_mealy, mealy_machine)
    w_method_oracle = WMethodEqOracleMealy(input_al, sul_mealy, 2, add_to_tree=True)
    # state_prefix_oracle = StatePrefixEqOracle(input_al, sul_mealy, 50, 100)

    L_sharp = Lsharp(input_al, sul_mealy, w_method_oracle, extension_rule=extension_rule, separation_rule=separation_rule, seed=seed, max_learning_rounds=75)
    learned_automaton, results, learning_rounds = L_sharp.run_Lsharp()

tests = [("Nothing", "SepSeq"), ("SepSeq", "SepSeq"), ("ADS", "SepSeq"), ("Nothing", "ADS"), ("SepSeq", "ADS"), ("ADS", "ADS")]

all_models = ["ASN_learnresult_SecureCode Aut_fix", "1_learnresult_MasterCard_fix", "LoesTarget", "Rabo_learnresult_SecureCode_Aut_fix", 
            "Rabo_learnresult_MAESTRO_fix", "ASN_learnresult_MAESTRO_fix", "4_learnresult_MAESTRO_fix", "10_learnresult_MasterCard_fix", 
            "Volksbank_learnresult_MAESTRO_fix", "learnresult_fix", "TCP_FreeBSD_Client", "TCP_Windows8_Client", "TCP_Linux_Client", 
            "DropBear", "OpenSSH", "TCP_Windows8_Server", "TCP_FreeBSD_Server", "TCP_Linux_Server", "BitVise",
            "OpenSSL_1.0.2_client_regular", "OpenSSL_1.0.1j_client_regular", "RSA_BSAFE_Java_6.1.1_server_regular",
            "miTLS_0.1.3_server_regular", "OpenSSL_1.0.2_server_regular", "NSS_3.17.4_client_regular", "GnuTLS_3.3.12_server_regular",
            "GnuTLS_3.3.12_client_regular", "NSS_3.17.4_server_regular", "OpenSSL_1.0.1l_server_regular", "OpenSSL_1.0.1g_client_regular", 
            "RSA_BSAFE_C_4.0.4_server_regular", "OpenSSL_1.0.1j_server_regular", "GnuTLS_3.3.8_client_regular", "OpenSSL_1.0.2_client_full", 
            "GnuTLS_3.3.8_server_regular", "GnuTLS_3.3.12_server_full", "GnuTLS_3.3.12_client_full", "OpenSSL_1.0.1g_server_regular", 
            "NSS_3.17.4_client_full", "GnuTLS_3.3.8_server_full", "GnuTLS_3.3.8_client_full"]
          
seeds = [
    81, 100, 158, 216, 245, 359, 366, 470, 560, 578, 580, 597, 661, 689, 692, 783, 818, 879, 930, 968,
    995, 1004, 1005, 1190, 1205, 1257, 1320, 1534, 1541, 1596, 1607, 1665, 1836, 1989, 2015, 2143, 2147, 2199, 2221, 2263,
    2283, 2365, 2370, 2408, 2495, 2528, 2554, 2558, 2561, 2588, 2610, 2619, 2679, 2773, 2816, 2950, 2966, 2969, 2983, 3044,
    3101, 3131, 3147, 3169, 3209, 3211, 3213, 3235, 3265, 3350, 3383, 3415, 3444, 3496, 3528, 3588, 3658, 3743, 3769, 3806,
    3809, 3900, 3980, 4094, 4179, 4358, 4370, 4447, 4467, 4535, 4550, 4588, 4632, 4646, 4689, 4782, 4845, 4948, 5102, 5409]

for dot_file in all_models:
    for (extension_rule, separation_rule) in tests:
        for seed in seeds:
            execution_time_new = timeit.timeit(lambda: benchmark(dot_file, extension_rule, separation_rule, seed), number=1)
            with open(file_path,mode="a",newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=fields)
                    writer.writerow({"model": dot_file, 
                                    "number_of_states": len(mealy_machine.states), 
                                    "number_of_inputs": len(input_al), 
                                    "complexity": len(input_al) * len(mealy_machine.states),
                                    "learning_rounds": learning_rounds, 
                                    "learn_resets": results[0], 
                                    "learn_steps": results[1], 
                                    "test_resets": results[2], 
                                    "test_steps": results[3], 
                                    "extension_rule": extension_rule, 
                                    "separation_rule": separation_rule, 
                                    "time": execution_time_new,
                                    "seed": seed
                    })