from aalpy.learning_algs import run_Lstar
from aalpy.utils import load_automaton_from_file
from aalpy.SULs import MealySUL
from aalpy.oracles import PerfectKnowledgeEqOracle
from WMethodEqOracleMealy import WMethodEqOracleMealy
import timeit
import csv, os

result_sul = 0
learned_automaton = None

fields = ["model", "number_of_states", "number_of_inputs", "complexity", "learning_rounds", "learn_queries", "learn_steps", "test_queries", "test_steps", "time"]
folder = "Experiment Results"
result_file = "Experiment3 Lstar.csv"
file_path = os.path.join(folder, result_file)


if not os.path.exists(file_path):
    with open(file_path,mode="w",newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

mealy_machine = None
learned_automaton = None
input_al = None
data = None

def benchmark(dot_file):
    global mealy_machine, learned_automaton, input_al, data
    dot_file = f'Lsharp/DotFiles/{dot_file}.dot'

    mealy_machine = load_automaton_from_file(dot_file, automaton_type='mealy')
    input_al = mealy_machine.get_input_alphabet()

    sul_mealy = MealySUL(mealy_machine)

    # perfect_oracle = PerfectKnowledgeEqOracle(input_al, sul_mealy, mealy_machine)
    w_method_oracle = WMethodEqOracleMealy(input_al, sul_mealy, 2, add_to_tree=False)
    # state_prefix_oracle = StatePrefixEqOracle(input_al, sul_mealy, 50, 100)

    learned_automaton, data = run_Lstar(input_al, sul_mealy, w_method_oracle, automaton_type='mealy', cache_and_non_det_check=False, cex_processing='rs', return_data=True, print_level=0)

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
          
for dot_file in all_models:
    for index in range(100):
        execution_time_new = timeit.timeit(lambda: benchmark(dot_file), number=1)
        with open(file_path,mode="a",newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writerow({"model": dot_file, 
                                "number_of_states": len(mealy_machine.states), 
                                "number_of_inputs": len(input_al), 
                                "complexity": len(input_al) * len(mealy_machine.states),
                                "learning_rounds": data['learning_rounds'], 
                                "learn_queries": data['queries_learning'], 
                                "learn_steps": data['steps_learning'], 
                                "test_queries": data['queries_eq_oracle'], 
                                "test_steps": data['steps_eq_oracle'], 
                                "time": execution_time_new
                })
