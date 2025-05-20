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
        if "retrieve_time" in data.keys():
            metrics["retrieve_time"].append(
                (float(data["retrieve_time"])/len(data["generation"])))
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
        """Save results to CSV file"""
        with open('results.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(headers)

            all_averages = defaultdict(list)
            unseen_averages = defaultdict(list)

            for category, category_data in all_results.items():
                category_averages = defaultdict(list)

                for dataset, metrics in category_data.items():
                    values = [category, dataset] + [
                        self.safe_mean(
                            metrics[h]) * (100 if "time" not in h and "length" not in h and "state_num" not in h else 1)
                        for h in headers[2:]
                    ]
                    csvwriter.writerow(values)

                    for key, value in zip(headers[2:], values[2:]):
                        category_averages[key].append(value)

                if category != "unseen":
                    for key in headers[2:]:
                        all_averages[key].append(
                            self.safe_mean(category_averages[key]))
                else:
                    for key in headers[2:]:
                        unseen_averages[key].append(
                            self.safe_mean(category_averages[key]))

                avg_values = [f"{category} Average", ""] + [
                    self.safe_mean(category_averages[key]) for key in headers[2:]
                ]
                csvwriter.writerow(avg_values)
                csvwriter.writerow([])

            final_all_averages = {
                key: self.safe_mean(all_averages[key]) for key in headers[2:]
            }
            overall_avg_values = ["Overall Average", ""] + [
                final_all_averages[key] for key in headers[2:]
            ]
            csvwriter.writerow(overall_avg_values)

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

    def analyze_multiple_dirs(self, result_dirs):
        category = ["nlu", "reasoning", "knowledge", "math", "safety"]
        
        aggregate_results = self.nested_dict()
        chosen_states_nums = self.nested_dict()
        time_dict = self.nested_dict()
        used_states = []
        
        for results_dir in result_dirs:
            eval_results, all_averages, _, all_results, used_states_list = self.analyze_results(
                results_dir)
            used_states += used_states_list

            # process chosen states numbers 
            self._process_chosen_states(
                category, chosen_states_nums, all_averages)

            # process time
            self._process_time_list(category, time_dict, all_averages)
               
            # process results
            self._process_results(eval_results, aggregate_results,
                                  category, all_averages, all_results)    
               
        # generate results report
        count_states = Counter(used_states)
        print(count_states)
        df = pd.DataFrame.from_dict(count_states, orient='index', columns=['count'])
        df.index = df.index.str.replace('_16shots', '')
        df = df.sort_values('count', ascending=True)

        plt.figure(figsize=(12, 8))
        plt.barh(df.index, df['count'])
        plt.xlabel('Usage Frequency', fontsize=16)
        plt.ylabel('Type of Task Vector', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=12)

        plt.title('Usage Frequency Distribution of different task vectors', fontsize=20, pad=20)

        # add numerical labels
        for i, v in enumerate(df['count']):
            plt.text(v, i, f' {v}', va='center')

        plt.tight_layout()
        plt.show()
        plt.savefig("used_states_distribution.png")
        self._generate_results_report(aggregate_results, chosen_states_nums, time_dict)

    def _process_time_list(self, category, time_dict, all_averages):
        for i, c in enumerate(category):
            time_dict[c]["clean_time"].append(all_averages["0shot_time"][i])
            time_dict[c]["intervene_time"].append(
                all_averages["intervene_time"][i])
            if "retrieve_time" in all_averages.keys():
                time_dict[c]["retrieve_time"].append(
                    all_averages["retrieve_time"][i])
            if len(all_averages["bm25_time"])>0 and len(all_averages[f"{nshots}shot_time"])>0:
                time_dict[c]["bm25_infer_time"].append(all_averages["bm25_time"][i])
                time_dict[c][f"{nshots}shot_time"].append(all_averages[f"{nshots}shot_time"][i])

    def _process_chosen_states(self, category, chosen_states_nums, all_averages):
        """process chosen states statistics"""
        for i, c in enumerate(category):
            chosen_states_nums[c]["chosen_nums"].append(
                all_averages["chosen_state_num"][i])

    def _process_results(self, eval_results, aggregate_results, category, 
                         all_averages, all_results):
        """process results - Ours only"""
        # process length data - Ours only
        aggregate_results["Length"]["Ours"].append(eval_results['zs_length'])
        
        # process results - Ours only
        for i, c in enumerate(category):
            # Ours only
            aggregate_results[c]["Ours"].append(all_averages['test_intervene_acc'][i])

        # process average value - Ours only
        aggregate_results["avg"]["Ours"].append(eval_results['test_intervene_acc'])

    def _generate_results_report(self, aggregate_results, chosen_states_nums, time_dict):
        """generate results report - Ours only"""
        # create dataframe and format
        chosen_num_df = pd.DataFrame(chosen_states_nums)
        
        # create dataframe - Ours only
        ours_results = {k: v for k, v in aggregate_results.items() if k == "Length" or isinstance(v, dict)}
        for category in ours_results:
            if isinstance(ours_results[category], dict):
                ours_results[category] = {"Ours": ours_results[category].get("Ours", [])}
        
        df = pd.DataFrame(ours_results)
        
        # process time information
        time_df = pd.DataFrame(time_dict).T

        # format function
        def format_cell(cell, digit_num=1):
            if isinstance(cell, list):
                mean = np.mean(cell)
                std = np.std(cell)
                return f"{mean:.{digit_num}f} ± {std:.{digit_num}f}"
            return str(cell)

        # apply formatting
        df_formatted = df.applymap(format_cell)
        chosen_num_df_formatted = chosen_num_df.applymap(format_cell)
        time_df_formatted = time_df.applymap(format_cell, digit_num=3)

        # print results
        print(chosen_num_df_formatted)
        print("\n")
        print(time_df_formatted)
        print("\n")
        print(df_formatted)

        # save results
        df_formatted.to_csv("csvs/results_ours_only.csv")
        chosen_num_df_formatted.to_csv("csvs/chosen_num.csv")
        time_df_formatted.to_csv("csvs/time.csv")

    def _plot_performance_distribution(self, aggregate_results, model):
        """plot performance distribution"""
        category = ["math", "nlu", "reasoning", "knowledge", "safety"]
        pre_performance_list = []
        post_performance_list = []

        for c in category:
            pre_performance_list.append(sum(aggregate_results[c]['zs'])/3)
            post_performance_list.append(sum(aggregate_results[c]['Ours'])/3)

        self._create_performance_plot(
            pre_performance_list, post_performance_list, model)

    def _create_performance_plot(self, pre_performance_list, post_performance_list, model):
        """create performance comparison plot"""
        category = ['Math', 'NLU', 'Reasoning', 'Knowledge', 'Safety']
        bar_width = 0.4
        r1 = np.arange(len(category))
        r2 = [x + bar_width for x in r1]

        fig, ax = plt.subplots(figsize=(23, 23))
        ax.bar(r1, pre_performance_list, color='skyblue',
               width=bar_width, label='Zero-shot')
        ax.bar(r2, post_performance_list, color='slateblue',
               width=bar_width, label='ELICIT')

        self._format_performance_plot(ax, category, bar_width,
                                      pre_performance_list, post_performance_list)
        plt.savefig(f"plots/performance_distribution_{model}.png")
        plt.close()

    def _format_performance_plot(self, ax, category, bar_width,
                                 pre_performance_list, post_performance_list):
        """format performance plot"""
        ax.set_ylabel('Accuracy', fontweight='bold', fontsize=52)
        plt.xticks(fontsize=52)
        plt.yticks(fontsize=52)
        ax.set_xticks([r + bar_width/2 for r in range(len(category))])
        ax.set_xticklabels([c.capitalize() for c in category])
        ax.legend(fontsize=52)

        self._add_value_labels(ax)
        self._highlight_math_difference(ax, bar_width,
                                        pre_performance_list[0], post_performance_list[0])
        plt.tight_layout()

    @staticmethod
    def _add_value_labels(ax, spacing=5):
        """add numerical labels to bar chart"""
        for rect in ax.patches:
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2
            space = spacing if y_value >= 0 else -spacing
            va = 'bottom' if y_value >= 0 else 'top'

            ax.annotate(
                f"{y_value:.1f}",
                (x_value, y_value),
                xytext=(0, space),
                textcoords="offset points",
                ha='center',
                va=va,
                fontsize=45
            )

    @staticmethod
    def _highlight_math_difference(ax, bar_width, math_pre, math_post):
        """highlight the difference in the math category"""
        math_diff = math_post - math_pre
        rect = plt.Rectangle(
            (bar_width/2, math_pre),
            bar_width,
            math_diff,
            fill=False,
            edgecolor='red',
            linestyle='--',
            linewidth=10
        )
        ax.add_patch(rect)


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
        "./eval_results/training_ATV/adapICV_top1_0.001_100_1e-5_15",
        "./eval_results/training_ATV/adapICV_top1_0.001_50_1e-5_15",
        "./eval_results/training_ATV/adapICV_top1_0.001_10_1e-5_15"
    ]

    analyzer.analyze_multiple_dirs(result_dirs)


if __name__ == "__main__":
    main()
