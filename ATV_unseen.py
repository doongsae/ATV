import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import csv
import sys
import seaborn as sns
import string
import re
from collections import Counter
nshots=16
need_reeval = False
class ResultAnalyzer:
    def __init__(self, weight_fv=None, retrieve_method="cosine"):
        self.weight_fv = weight_fv
        self.retrieve_method = retrieve_method
        self.datasets = {
            'nlu': ["superglue_rte", "superglue_wic",  "glue_qnli", "glue_sst2", "glue_mnli"], #,
            'reasoning': ["arc_challenge", "bbh_boolean_expressions", "bbh_date_understanding",
                          "bbh_reasoning_about_colored_objects", "bbh_temporal_sequences"],
            'knowledge': ["boolq", "commonsense_qa", "hellaswag", "openbookqa"],
            'math': ["math_qa", "mmlu_pro_math"],
            'safety': ["bbq_age", "crows_pairs", "ethics_justice", "ethics_commonsense"], #
            'unseen': ["glue_cola", "bbq_religion", "deepmind",
                       "mmlu_high_school_psychology", "bbh_logical_deduction_five_objects"]
            # 'math': ["math_qa", "mmlu_pro_math"]
        }

    @staticmethod
    def nested_dict():
        return defaultdict(lambda: defaultdict(list))

    @staticmethod
    def safe_mean(arr):
        """Calculate mean of numeric values, ignoring NaN"""
        return float(np.mean([x for x in arr if isinstance(x, (int, float)) and not np.isnan(x)]))

    def _get_filtered_files(self, results_dir, recall):
        """Get relevant files based on filtering criteria"""
        all_files = os.listdir(results_dir)
        file_list = [f for f in all_files if all(x in f for x in
                [self.retrieve_method, f"{self.weight_fv}fv", f"{recall}recall"])]
        return [f for f in file_list if "parsed" not in f]
    
    def get_answer_type(self, answer):
        if answer in string.ascii_uppercase:
            return 'capital'
        elif answer in ['True', 'False', 'Neither']:
            return 'true_false'
        elif answer in ['Yes', 'No']:
            return 'yes_no'
        elif answer in ['positive', 'negative']:
            return 'positive_negative'
        elif answer.isdigit():
            return 'number'
        return None
    
    def find_answer(self, text, answer):
        # arc chanllenge, bbh date understanding, bbh reasoning about colored, bbh temporal sequences,  bbq age, bbq religion, commonsenseqa, crows_pairs, deepmind, hellaswag, mathqa, mmlu high school psychology, MMLU pro math, openbook qa: ABC
        # bbh boolean, boolq, superglue rte: true/false
        # ethics commonsense, ethics justice, glue cola, glue qnli, superglue wic: Yes/No
        # glue mnli : True False Neither
        # glue sst2: positive / negative
        patterns = {
        'capital': r'([A-Z])(?:\.|:|<\|eot_id\|>)?\b',
        'true_false': r'\b(True|False|Neither)(?:\.|:|<\|eot_id\|>)?\b', 
        'number': r'(\d+)(?:\.|:|<\|eot_id\|>)?\b',
        'yes_no': r'\b(Yes|No)(?:\.|:|<\|eot_id\|>)?\b',
        'positive_negative': r'\b(positive|negative|Positive|Negative)(?:\.|:|<\|eot_id\|>)?\b'
        }
        # Strip the expected answer
        answer = answer.strip()
        pattern_type = self.get_answer_type(answer)
        if pattern_type == None: breakpoint()
        assert pattern_type is not None
        pattern = patterns[pattern_type]
        
        text = text.replace("X<|eot_id|>", "")
        if "answer" in text:
            p = text.split("answer")[-1]
            # breakpoint()
            
            for pattern in patterns.values():
                matches = re.findall(pattern, p)
                if len(matches) > 0:
                    pred_answer = matches[0].strip().replace("<|eot_id|>", "").strip(".").strip(":").lower()
                    if matches and  pred_answer== answer.lower():
                        return 1, pred_answer
                
        # Check each pattern
       
        matches = re.findall(pattern, text)
            
        if len(matches) > 0:
            pred_answer = matches[-1].strip().replace("<|eot_id|>", "").strip(".").strip(":").lower()
            if matches and  pred_answer== answer.lower():
                return 1, pred_answer
        try:
            return 0, pred_answer
        except:
            return 0, None
    
    def reeval(self, item_list, dataset = None):
        zs_acc_list = []
        intervene_acc_list = []
        pred_results = []
        for g in item_list["generation"]:
          
           zs_acc, pred_answer = self.find_answer(g["clean_output"], g["label"])
           
           zs_acc_list.append(zs_acc)
           intervene_acc, intervene_pred_answer = self.find_answer(g["intervene_output"], g["label"])
           intervene_acc_list.append(intervene_acc)
           item = {
               "clean_output": g["clean_output"],
               "clean_parsed_output": pred_answer,
                "zs_acc": zs_acc,
               "intervene_output": g["intervene_output"],
               "intervene_parsed_answer": intervene_pred_answer,
               "label": g["label"].strip(),
               "intervene_acc": intervene_acc
               
           }
           pred_results.append(item)

        return sum(zs_acc_list) / len(zs_acc_list), sum(intervene_acc_list)/len(intervene_acc_list), pred_results

    
    def _process_data_entry(self, data, all_results, category, dataset_name, results_dir):
        """Process a single data entry and update results dictionary"""
        if 'test_zero-shot_acc' not in data.keys():
            return
        metrics = all_results[category][dataset_name]

        # Process test accuracy
        if need_reeval:
            zs_acc, intervene_acc, pred_results = self.reeval(data, dataset_name)
        else:
            zs_acc, intervene_acc = float(data['test_zero-shot_acc']), (float(data['test_acc'][str(data["icl_best_layer"])])
                    if isinstance(data['test_acc'], dict)
                    else float(data['test_acc']))
        
        metrics["test_zero_acc"].append(zs_acc)

        metrics["test_intervene_acc"].append(intervene_acc)
        metrics["harm"].append(int(intervene_acc < zs_acc))
        metrics["retrieve_acc"].append(float(data['retrieve_acc']))

        # Process timing and length metrics
        metrics["0shot_time"].append(
            (float(data['clean_time'])/len(data["generation"])))
        metrics["intervene_time"].append(
            (float(data['intervene_time'])/len(data["generation"])))
        # if "retrieve_time" in data.keys():
        #     metrics["retrieve_time"].append(
        #         (float(data["retrieve_time"])/len(data["generation"])))
        metrics["zs_length"].append(float(data['zs_lengths']))

        # Calculate chosen state numbers
        state_num = sum(len(item["chosen_states"])
                        for item in data["generation"])
        metrics["chosen_state_num"].append(
            float(state_num/len(data["generation"])))
        
        # Process BM25 results if available
    
        # print("valid" not in results_dir and "bm25_results" in data.keys() and data["bm25_results"] != {})
        if "valid" not in results_dir and "bm25_results" in data.keys() and data["bm25_results"] != {}:
            bm25_results = data["bm25_results"]
            if need_reeval:
                zs_acc, intervene_acc, bm25_pred_results = self.reeval(bm25_results)
            else:
                zs_acc, intervene_acc = float(bm25_results["acc"]), float(bm25_results["+tv_acc"])
            metrics["bm25_acc"].append(zs_acc)
            metrics["bm25_length"].append(float(bm25_results["length"]))
            metrics["bm25_time"].append(
                float(bm25_results["time"])/len(data["generation"]))
            metrics["bm25_tv"].append(intervene_acc)

        # Process nshots results
       
        if "nshots_results" in data.keys() and data["nshots_results"][f"{nshots}shot_results"] != {}:
            for key in data["nshots_results"].keys():
                if need_reeval:
                    zs_acc, intervene_acc, nshot_pred_answer = self.reeval((data["nshots_results"][key]))
                else:
                    zs_acc, intervene_acc = float(data["nshots_results"][key]["acc"]), float(data["nshots_results"][key]["+tv_acc"])
                all_results[category][dataset_name][f"{key.split('_')[0]}_acc"].append(zs_acc)
                all_results[category][dataset_name][f"{key.split('_')[0]}_tv"].append(intervene_acc)
                all_results[category][dataset_name][f"{key.split('_')[0]}_length"].append(
                    float(data["nshots_results"][key]["length"]))
                all_results[category][dataset_name][f"{key.split('_')[0]}_time"].append(
                    float(data["nshots_results"][key]["time"])/len(data["generation"]))
        
        used_states = []
        for g in data["generation"]:
            used_states+=g["chosen_states"]
        if need_reeval: return pred_results, used_states
        else: return None, used_states

    def _format_results_table(self, all_results):
        """Format results into a readable table"""
        headers = ["Category", "dataset"] + list(
            all_results[next(iter(all_results))][
                next(iter(all_results[next(iter(all_results))]))
            ].keys()
        )
        widths = [15, 30] + [15] * (len(headers) - 2)

        return headers, widths

    def _save_results_csv(self, headers, widths, all_results):
        all_averages = defaultdict(list)
        unseen_averages = defaultdict(list)

        for category, category_data in all_results.items():
            category_averages = defaultdict(list)

            for dataset, metrics in category_data.items():
                values = [category, dataset] + [
                    self.safe_mean(metrics[h]) * (100 if "time" not in h and "length" not in h and "state_num" not in h else 1)
                    for h in headers[2:]
                ]

                for key, value in zip(headers[2:], values[2:]):
                    category_averages[key].append(value)

            if category != "unseen":
                for key in headers[2:]:
                    all_averages[key].append(self.safe_mean(category_averages[key]))
            else:
                for key in headers[2:]:
                    unseen_averages[key].append(self.safe_mean(category_averages[key]))

        final_all_averages = { key: self.safe_mean(all_averages[key]) for key in headers[2:] }

        return final_all_averages, all_averages, unseen_averages


    def _plot_layer_distribution(self, valid_best_layers):
        """Plot distribution of best layers"""
        bins = np.arange(0, 33, 1)
        plt.figure(figsize=(10, 6))
        plt.hist(valid_best_layers, bins=bins, edgecolor='black')
        plt.title('Best Layer Distribution', fontsize=16)
        plt.xlabel('Layer', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.xticks(bins)
        plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
        plt.xlim(0, 32)
        plt.grid(True)
        plt.savefig("plots/best_layer_distribution_valid.png")
        plt.close()

    def analyze_results(self, results_dir, recall=0.2, tweet=False):
        """Main analysis function that processes all results"""
        all_results = defaultdict(self.nested_dict)
        valid_best_layers = []
        filtered_files = self._get_filtered_files(results_dir, recall)
        used_states_list = []
        # Process each dataset
        for category, dataset_list in self.datasets.items():
            for dataset_name in dataset_list:
                if tweet:
                    dataset_name = f"{dataset_name}_tweet"

                try:
                    file_name = next(
                        f for f in filtered_files if dataset_name in f)
                except StopIteration:
                    print(f"No file found for dataset: {dataset_name}")
                    continue
                
                with open(os.path.join(results_dir, file_name), "r") as f:
                    all_data = json.load(f)
                all_pred_results = []
                for data in all_data:
                    pred_results, used_states = self._process_data_entry(
                        data, all_results, category, dataset_name, results_dir)
                    all_pred_results.append(pred_results)
                    used_states_list += used_states
                    if 'valid_best_layer' in data:
                        valid_best_layers.append(data['valid_best_layer'])
                with open(os.path.join(results_dir, file_name.replace(".json", "parsed.json")), "w") as f:
                    json.dump(all_pred_results, f, indent=4)

        # Format and save results
        headers, widths = self._format_results_table(all_results)
        final_averages, all_averages, unseen_averages = self._save_results_csv(
            headers, widths, all_results
        )

        # Plot distribution
        # self._plot_layer_distribution(valid_best_layers)

        return final_averages, all_averages, unseen_averages, all_results, used_states_list

    def plot_recall_curve(self, results_dir, model="mamba"):
        """Plot accuracy vs recall curve"""
        recalls = [0.2, 0.4, 0.6, 0.8, 1.0]
        accs = []

        for recall in recalls:
            eval_results, _, _, _ = self.analyze_results(
                results_dir, recall=recall)
            accs.append(eval_results['test_intervene_acc'])
            baseline = eval_results["test_zero_acc"]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(recalls, accs, 'r-', linewidth=2, label='Intervene Accuracy')
        ax.axhline(baseline, linestyle='--', linewidth=2,
                   label='Zero-Shot Accuracy')

        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Accuracy vs. Recall', fontsize=14)
        ax.set_xlim(0.1, 1.1)
        ax.legend(loc='lower right', fontsize=10)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='out', length=6, width=2)
        ax.grid(color='gray', linestyle='--', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(f"plots/{model}_recall_vs_acc.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _process_unseen_time_list(self, unseen, unseen_time_dict, all_results):
        """Process time statistics for unseen datasets"""
        for c in unseen:
            unseen_time_dict[c]["clean_time"] = all_results["unseen"][c]["0shot_time"]
            unseen_time_dict[c]["intervene_time"] = all_results["unseen"][c]["intervene_time"]
            if "retrieve_time" in all_results["unseen"][c]:
                unseen_time_dict[c]["retrieve_time"] = all_results["unseen"][c]["retrieve_time"]
            if len(all_results["unseen"][c]["bm25_time"]) > 0 and f"{nshots}shot_time" in all_results["unseen"][c]:
                unseen_time_dict[c]["bm25_infer_time"] = all_results["unseen"][c]["bm25_time"]
                unseen_time_dict[c][f"{nshots}shot_time"] = all_results["unseen"][c][f"{nshots}shot_time"]

    def _process_unseen_chosen_states(self, unseen, unseen_chosen_states_nums, all_results):
        """Process chosen states statistics for unseen datasets"""
        for c in unseen:
            unseen_chosen_states_nums[c]["chosen_nums"] = all_results['unseen'][c]["chosen_state_num"]

    def analyze_multiple_dirs(self, result_dirs):
        """Analyze multiple result directories for unseen datasets"""
        unseen = ["glue_cola", "bbq_religion", "deepmind",
                  "mmlu_high_school_psychology", "bbh_logical_deduction_five_objects"]
        
        unseen_results = self.nested_dict()
        unseen_chosen_states_nums = self.nested_dict()
        unseen_time_dict = self.nested_dict()
        used_states = []
        
        for results_dir in result_dirs:
            _, _, unseen_averages, all_results, used_states_list = self.analyze_results(
                results_dir)
            used_states += used_states_list

            # process chosen states numbers for unseen datasets
            self._process_unseen_chosen_states(
                unseen, unseen_chosen_states_nums, all_results)

            # process time for unseen datasets
            self._process_unseen_time_list(
                unseen, unseen_time_dict, all_results)
               
            # process results for unseen datasets
            self._process_results(unseen_results, unseen, unseen_averages, all_results)    
               
        # generate results report
        count_states = Counter(used_states)
        # print(count_states)
        df = pd.DataFrame.from_dict(count_states, orient='index', columns=['count'])
        df.index = df.index.str.replace('_16shots', '')
        df = df.sort_values('count', ascending=True)

        plt.figure(figsize=(12, 8))
        plt.barh(df.index, df['count'])
        plt.xlabel('Usage Frequency', fontsize=16)
        plt.ylabel('Type of Task Vector', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=12)

        plt.title('Usage Frequency Distribution (Unseen Datasets)', fontsize=20, pad=20)

        # add numerical labels
        for i, v in enumerate(df['count']):
            plt.text(v, i, f' {v}', va='center')

        plt.tight_layout()
        plt.show()
        # plt.savefig("used_states_distribution_unseen.png")
        self._generate_results_report(unseen_results, unseen_chosen_states_nums, unseen_time_dict)

    def _process_results(self, unseen_results, unseen, unseen_averages, all_results):
        """Process results for unseen datasets - including only Ours and zs"""
        # Process results for each unseen dataset
        for u in unseen:
            # Length data
            if "zs" not in unseen_results or "Length" not in unseen_results["zs"]:
                unseen_results["zs"]["Length"] = []
                unseen_results["Ours"]["Length"] = []
            
            if len(all_results['unseen'][u]["zs_length"]) > 0:
                unseen_results["zs"]["Length"].append(
                    sum(all_results['unseen'][u]["zs_length"]) /
                    len(all_results['unseen'][u]["zs_length"])
                )
                unseen_results["Ours"]["Length"].append(
                    sum(all_results['unseen'][u]["zs_length"]) /
                    len(all_results['unseen'][u]["zs_length"])
                )
            
            # Save results for each dataset
            if u not in unseen_results["zs"]:
                unseen_results["zs"][u] = []
                unseen_results["Ours"][u] = []
            
            # Accuracy data
            if len(all_results['unseen'][u]["test_zero_acc"]) > 0:
                unseen_results["zs"][u].append(
                    sum(all_results['unseen'][u]["test_zero_acc"]) /
                    len(all_results['unseen'][u]["test_zero_acc"]) * 100
                )
                unseen_results["Ours"][u].append(
                    sum(all_results['unseen'][u]["test_intervene_acc"]) /
                    len(all_results['unseen'][u]["test_intervene_acc"]) * 100
                )
        
        # Calculate avg values directly from dataset averages
        if "avg" not in unseen_results["zs"]:
            unseen_results["zs"]["avg"] = []
            unseen_results["Ours"]["avg"] = []
        
        # Calculate average of all datasets in current directory
        zs_avg = []
        ours_avg = []
        for u in unseen:
            if u in unseen_results["zs"] and len(unseen_results["zs"][u]) > 0:
                zs_avg.append(unseen_results["zs"][u][-1])  # Most recently added value
            
            if u in unseen_results["Ours"] and len(unseen_results["Ours"][u]) > 0:
                ours_avg.append(unseen_results["Ours"][u][-1])  # Most recently added value
        
        if zs_avg:
            unseen_results['zs']['avg'].append(sum(zs_avg) / len(zs_avg))
        
        if ours_avg:
            unseen_results['Ours']['avg'].append(sum(ours_avg) / len(ours_avg))

    def _generate_results_report(self, unseen_results, unseen_chosen_states_nums, unseen_time_dict):
        """Generate results report for unseen datasets - including only Ours and zs"""
        # Create and format dataframes
        unseen_chosen_num_df = pd.DataFrame(unseen_chosen_states_nums)
        
        # Filter results to include only Ours
        filtered_results = {"Ours": unseen_results["Ours"]}
        
        unseen_df = pd.DataFrame(filtered_results).T
        unseen_time_df = pd.DataFrame(unseen_time_dict).T

        # Formatting function
        def format_cell(cell, digit_num=1):
            if isinstance(cell, list):
                mean = np.mean(cell)
                std = np.std(cell)
                return f"{mean:.{digit_num}f} ± {std:.{digit_num}f}"
            return f"{cell:.{digit_num}f}" if isinstance(cell, (int, float)) else str(cell)

        # Apply formatting
        unseen_df_formatted = unseen_df.map(format_cell)
        unseen_num_df_formatted = unseen_chosen_num_df.map(format_cell)
        unseen_time_df_formatted = unseen_time_df.map(format_cell, digit_num=3)

        # Print results
        print("[Unseen Dataset Results]")
        # print("\nChosen States Numbers:")
        # print(unseen_num_df_formatted)
        print("\n[Time Statistics]")
        print(unseen_time_df_formatted)
        print("\n[Accuracy Results]")
        print(unseen_df_formatted)

def main():
    """main function"""
    # # weight fv
    # model = sys.argv[2]
    # suffix = sys.argv[3]

    weight_fv = 0.001
    lr = "5e-4"
    epochs = 15

    analyzer = ResultAnalyzer(weight_fv=weight_fv)
    model = "llama3"
    result_dirs = [
        "eval_results/evaluate_ATV/adapICV_top1_0.001_42_1e-5_15",
        # "eval_results/evaluate_ATV/adapICV_top1_0.001_100_1e-5_15",
        # "eval_results/evaluate_ATV/adapICV_top1_0.001_10_1e-5_15",
    ]

    analyzer.analyze_multiple_dirs(result_dirs)


if __name__ == "__main__":
    main()
