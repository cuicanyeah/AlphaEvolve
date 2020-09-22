// Copyright 2020 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "evaluator.h"

#include <algorithm>
#include <iomanip>
#include <ios>
#include <limits>
#include <memory>
#include <string>
#include <fstream>
#include <queue>
#include <cmath>
#include <stdlib.h> 

#include "task.h"
#include "task_util.h"
#include "task.pb.h"
#include "definitions.h"
#include "executor.h"
#include "random_generator.h"
#include "train_budget.h"
#include "google/protobuf/text_format.h"
#include "absl/algorithm/container.h"
#include "absl/flags/flag.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"

namespace automl_zero {

bool operatorf(std::pair<double, int> i, std::pair<double, int> j) { return (i.first > j.first);}

using ::absl::c_linear_search;  // NOLINT
using ::absl::GetFlag;  // NOLINT
using ::absl::make_unique;  // NOLINT
using ::std::cout;  // NOLINT
using ::std::endl;  // NOLINT
using ::std::fixed;  // NOLINT
using ::std::make_shared;  // NOLINT
using ::std::min;  // NOLINT
using ::std::mt19937;  // NOLINT
using ::std::nth_element;  // NOLINT
using ::std::pair;  // NOLINT
using ::std::setprecision;  // NOLINT
using ::std::vector;  // NOLINT
using ::std::unique_ptr;  // NOLINT
using internal::CombineFitnesses;

constexpr IntegerT kMinNumTrainExamples = 10;
constexpr RandomSeedT kFunctionalCacheRandomSeed = 235732282;

Evaluator::Evaluator(const FitnessCombinationMode fitness_combination_mode,
                     const TaskCollection& task_collection,
                     RandomGenerator* rand_gen,
                     FECCache* functional_cache,
                     TrainBudget* train_budget,
                     const double max_abs_error)
    : fitness_combination_mode_(fitness_combination_mode),
      task_collection_(task_collection),
      train_budget_(train_budget),
      rand_gen_(rand_gen),
      functional_cache_(functional_cache),
      functional_cache_bit_gen_owned_(
          make_unique<mt19937>(kFunctionalCacheRandomSeed)),
      functional_cache_rand_gen_owned_(
          make_unique<RandomGenerator>(functional_cache_bit_gen_owned_.get())),
      functional_cache_rand_gen_(functional_cache_rand_gen_owned_.get()),
      best_fitness_(-1.0),
      max_abs_error_(max_abs_error),
      num_train_steps_completed_(0) {
  FillTasks(task_collection_, &tasks_);
  CHECK_GT(tasks_.size(), 0);
}

// , vector<double>& ICs
std::pair<double, std::vector<double>> Evaluator::Evaluate(const Algorithm& algorithm, double best_select_fitness, IntegerT is_search, std::vector<double>* strategy_ret, std::vector<double>* valid_strategy_ret, std::vector<IntegerT>* useful_list) {
  // Compute the mean fitness across all tasks.
  vector<double> task_fitnesses;
  task_fitnesses.reserve(tasks_.size());
  vector<IntegerT> task_indexes;  // Tasks to use.
  vector<vector<double>> all_task_preds;
  vector<vector<double>> all_price_diff;  
  IntegerT num_stock_rank = 0;
  IntegerT num_TS_rank = 0;

  IntegerT num_stock_rank_count = 0;
  IntegerT num_TS_rank_count = 0;

  // vector<IntegerT> useful_list;
  vector<IntegerT> useful_list_predict_only;
  // vector<IntegerT> useful_list_true;
  useful_list->resize(algorithm.predict_.size() + algorithm.learn_.size());
  useful_list_predict_only.resize(algorithm.predict_.size());
  // useful_list_true.resize(algorithm.predict_.size() + algorithm.learn_.size());
  // cout << "add here 8333" << endl;
  std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>> check_cycle_list;
  std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>> check_cycle_list_predict_only;
  // cout << "add here 85555" << endl;
  // check_cycle_list->resize(algorithm.predict_.size() + algorithm.learn_.size() + 155);

  // cout << algorithm.ToReadable() << endl;
  // CHECK(num_stock_rank > 9);
  bool result = CheckHasIn(&algorithm, 0, 1, useful_list, &check_cycle_list, -1, 0); // james: initialize main pos as -1 since if as 0 then will skip first ins
  bool result_predict_only = CheckHasIn(&algorithm, 0, 1, &useful_list_predict_only, &check_cycle_list_predict_only, -1, 1);
  // bool result_true = CheckHasIn(&algorithm, 0, 1, &useful_list_true, &check_cycle_list, -1); 

  // 应该把这些反过来写，直接从 cycle check list里面loop，
    for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list.begin(); i != check_cycle_list.end(); ++i) {
      if ((i->second).second == 0) {
        (i->second).second = 3;
        // cout << "all recheck this type: " <<  (i->first).second << "(" << (i->first).first << ") in this " << " instruction_num:" << (i->second).first << " as: " << 0 << endl;   
        // ++debug_count;
        // CHECK(debug_count < 2);
        // ++ins_count_find;
        // if_continue = true;
        // if (CheckHasIn(&algorithm, (i->first).second, (i->first).first, &useful_list, &check_cycle_list, (i->second).first + 1))
        //  useful_list[(i->second).first] = 1; 
      }      
    }
    for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list.begin(); i != check_cycle_list.end(); ++i) {
      if ((i->second).second == 3) {
        // (i->second).second = 3;
        // cout << "all recheck this type: " <<  (i->first).second << "(" << (i->first).first << ") in this " << " instruction_num:" << (i->second).first << " as: " << (i->second).second << endl;   
        // ++debug_count;
        // if ((i->first).second == 1 && (i->first).first == 0) CHECK((i->first).second < -2);
        // ++ins_count_find;
        // if_continue = true;
        if (CheckHasIn(&algorithm, (i->first).second, (i->first).first, useful_list, &check_cycle_list, (i->second).first + 1, 0))
         (*useful_list)[(i->second).first] = 1; 
      }      
    }

    // james: below is to do the same for predict only since for prediction we need this dependency between instructions in predict function
    for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list_predict_only.begin(); i != check_cycle_list_predict_only.end(); ++i) {
      if ((i->second).second == 0) {
        (i->second).second = 3;
        // cout << "all recheck this type: " <<  (i->first).second << "(" << (i->first).first << ") in this " << " instruction_num:" << (i->second).first << " as: " << 0 << endl;   
        // ++debug_count;
        // CHECK(debug_count < 2);
        // ++ins_count_find;
        // if_continue = true;
        // if (CheckHasIn(&algorithm, (i->first).second, (i->first).first, &useful_list, &check_cycle_list, (i->second).first + 1))
        //  useful_list[(i->second).first] = 1; 
      }      
    }
    for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list_predict_only.begin(); i != check_cycle_list_predict_only.end(); ++i) {
      if ((i->second).second == 3) {
        // (i->second).second = 3;
        // cout << "all recheck this type: " <<  (i->first).second << "(" << (i->first).first << ") in this " << " instruction_num:" << (i->second).first << " as: " << (i->second).second << endl;   
        // ++debug_count;
        // if ((i->first).second == 1 && (i->first).first == 0) CHECK((i->first).second < -2);
        // ++ins_count_find;
        // if_continue = true;
        if (CheckHasIn(&algorithm, (i->first).second, (i->first).first, &useful_list_predict_only, &check_cycle_list_predict_only, (i->second).first + 1, 1))
         useful_list_predict_only[(i->second).first] = 1; 
      }      
    }

  IntegerT predict_only_count = 0;
  for(std::vector<IntegerT>::iterator i = useful_list_predict_only.begin(); i != useful_list_predict_only.end(); ++i) {
    if (*i == 1 && (*useful_list)[predict_only_count + algorithm.learn_.size()] == 0) (*useful_list)[predict_only_count + algorithm.learn_.size()] = 1;
    // cout << "useful_list:" << *i << endl;
    ++predict_only_count;
    // ++(*i);
  }
  // IntegerT instruction_num = 0;
  // IntegerT main_pos = -1;

  // vector<double> list_int_op_out = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,21,27,34,44,47,50,51,54,55,56,59,62,65,66,67,71,72,74,75};
  // vector<double> list_vec_op_out = {16,18,19,20,22,23,24,25,26,31,35,36,45,48,52,53,57,60,63,68,69,73};
  // vector<double> list_mat_op_out = {17,28,29,30,32,33,37,38,39,40,41,42,43,46,49,58,61,64};

  // for (IntegerT i: useful_list) {
  //   if (i != 1) {
  //     if (instruction_num >= algorithm.learn_.size()) {
  //       main_pos = instruction_num - algorithm.learn_.size();
  //       const std::shared_ptr<const Instruction>& myinstruction = algorithm.predict_[main_pos]; 

  //     if (myinstruction->op_ == 0) continue;

  //     bool found3 = (std::find(list_int_op_out.begin(), list_int_op_out.end(), myinstruction->op_) != list_int_op_out.end());
  //     bool found3_vec = (std::find(list_vec_op_out.begin(), list_vec_op_out.end(), myinstruction->op_) != list_vec_op_out.end());
  //     bool found3_mat = (std::find(list_mat_op_out.begin(), list_mat_op_out.end(), myinstruction->op_) != list_mat_op_out.end());
  //     cout << "here goes the recheck!!!!!!!!!" << endl;
  //     cout << "what is this num" << instruction_num << endl;
  //     bool if_continue = false;
  //     if (found3) {
  //       for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list.begin(); i != check_cycle_list.end(); ++i) {
  //         if ((i->first).first == myinstruction->out_ && (i->first).second == 0 && (i->second).first == (instruction_num + algorithm.learn_.size()) && (i->second).second == 0) {
  //           (i->second).second = 1;
  //           cout << "predict recheck this type: " << 0 << "(" << myinstruction->out_ << ") in this " << "instruction here: " << myinstruction->ToString() << " instruction_num:" << instruction_num << " as: s" << 0 << endl;  
  //           // ++debug_count;
  //           // CHECK(debug_count < 2);
  //           // ++ins_count_find;
  //           // if_continue = true;
  //         } else {
  //           if_continue = true;
  //           break;
  //         }      
  //       }        
  //       if (CheckHasIn(&algorithm, 0, myinstruction->out_, &useful_list, &check_cycle_list, instruction_num + algorithm.learn_.size() + 1))
  //       useful_list[instruction_num] = 1;
  //     } 
  //     if (found3_vec) {
  //       for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list.begin(); i != check_cycle_list.end(); ++i) {
  //         if ((i->first).first == myinstruction->out_ && (i->first).second == 1 && (i->second).first == (instruction_num + algorithm.learn_.size()) && (i->second).second == 0) {
  //           (i->second).second = 1;
  //           cout << "predict recheck this type: " << 0 << "(" << myinstruction->out_ << ") in this " << "instruction here: " << myinstruction->ToString() << " instruction_num:" << instruction_num << " as: v" << 1 << endl;   
  //           // ++debug_count;
  //           // CHECK(debug_count < 2);
  //           // ++ins_count_find;
  //           // if_continue = true;
  //         } else {
  //           if_continue = true;
  //           break;
  //         }      
  //       }
  //       if (CheckHasIn(&algorithm, 1, myinstruction->out_, &useful_list, &check_cycle_list, instruction_num + algorithm.learn_.size() + 1))
  //       useful_list[instruction_num] = 1;
  //     } 
  //     if (found3_mat)  {
  //       for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list.begin(); i != check_cycle_list.end(); ++i) {
  //         if ((i->first).first == myinstruction->out_ && (i->first).second == 2 && (i->second).first == (instruction_num + algorithm.learn_.size()) && (i->second).second == 0) {
  //           (i->second).second = 1;
  //           cout << "predict recheck this type: " << 0 << "(" << myinstruction->out_ << ") in this " << "instruction here: " << myinstruction->ToString() << " instruction_num:" << instruction_num << " as: m" << 2 << endl;  
  //           // ++debug_count;
  //           // CHECK(debug_count < 2);
  //           // ++ins_count_find;
  //           // if_continue = true;
  //         } else {
  //           if_continue = true;
  //           break;
  //         }      
  //       }
  //       if (CheckHasIn(&algorithm, 2, myinstruction->out_, &useful_list, &check_cycle_list, instruction_num + algorithm.learn_.size() + 1))
  //         useful_list[instruction_num] = 1;        
  //     }

  //     if (if_continue) continue;

  //     } else {
  //       cout << "here goes the recheck!!!!!!!!! learn!!!!" << endl;
  //       cout << "what is this num" << instruction_num << endl;
  //       // james: else is for predict part 
  //       main_pos = instruction_num;
  //       const std::shared_ptr<const Instruction>& myinstruction = algorithm.learn_[main_pos]; 
  //     cout << "heaviside here goes the recheck!!!!!!!!! predict!!!! 1" << endl;
  //     if (myinstruction->op_ == 0) continue; 
  //     cout << "heaviside here goes the recheck!!!!!!!!! predict!!!! 2" << endl;
  //     bool found3 = (std::find(list_int_op_out.begin(), list_int_op_out.end(), myinstruction->op_) != list_int_op_out.end());
  //     bool found3_vec = (std::find(list_vec_op_out.begin(), list_vec_op_out.end(), myinstruction->op_) != list_vec_op_out.end());
  //     bool found3_mat = (std::find(list_mat_op_out.begin(), list_mat_op_out.end(), myinstruction->op_) != list_mat_op_out.end());
  //     cout << "heaviside here goes the recheck!!!!!!!!! predict!!!! 3" << endl;
  //     bool if_continue = false;
  //     if (found3) {
  //       for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list.begin(); i != check_cycle_list.end(); ++i) {
  //         if ((i->first).first == myinstruction->out_ && (i->first).second == 0 && (i->second).first == instruction_num && (i->second).second == 0) {
  //           (i->second).second = 1;
  //           cout << "learn recheck this type: " << 0 << "(" << myinstruction->out_ << ") in this " << "instruction here: " << myinstruction->ToString() << " instruction_num:" << instruction_num << " as: s" << 0 << endl;   
  //           // ++debug_count;
  //           // CHECK(debug_count < 2);
  //           // ++ins_count_find;
  //           // if_continue = true;
  //         } else {
  //           if_continue = true;
  //           break;
  //         }
  //       }        
  //       if (CheckHasIn(&algorithm, 0, myinstruction->out_, &useful_list, &check_cycle_list, instruction_num + 1))
  //       useful_list[instruction_num] = 1;
  //     } 



  //     if (found3_vec) {
  //       for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list.begin(); i != check_cycle_list.end(); ++i) {
  //         if ((i->first).first == myinstruction->out_ && (i->first).second == 1 && (i->second).first == instruction_num && (i->second).second == 0) {
  //           (i->second).second = 1;
  //           cout << "learn recheck this type: " << 0 << "(" << myinstruction->out_ << ") in this " << "instruction here: " << myinstruction->ToString() << " instruction_num:" << instruction_num << " as: v" << 0 << endl;   
  //           // ++debug_count;
  //           // CHECK(debug_count < 2);
  //           // ++ins_count_find;
  //           // if_continue = true;
  //         } else {
  //           if_continue = true;
  //           break;
  //         }      
  //       }
  //       if (CheckHasIn(&algorithm, 1, myinstruction->out_, &useful_list, &check_cycle_list, instruction_num + 1))
  //        useful_list[instruction_num] = 1; 
  //     } 
  //     if (found3_mat) {
  //       for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list.begin(); i != check_cycle_list.end(); ++i) {
  //         if ((i->first).first == myinstruction->out_ && (i->first).second == 2 && (i->second).first == instruction_num && (i->second).second == 0) {
  //           (i->second).second = 1;
  //           cout << "learn recheck this type: " << 0 << "(" << myinstruction->out_ << ") in this " << "instruction here: " << myinstruction->ToString() << " instruction_num:" << instruction_num << " as: m" << 0 << endl;  
  //           // ++debug_count;
  //           // CHECK(debug_count < 2);
  //           // ++ins_count_find;
  //           // if_continue = true;
  //         } else {
  //           if_continue = true;
  //           break;
  //         }      
  //       }        
  //       if (CheckHasIn(&algorithm, 2, myinstruction->out_, &useful_list, &check_cycle_list, instruction_num + 1))
  //        useful_list[instruction_num] = 1;     
  //     }

  //     if (if_continue) continue;

  //     } 
  //   }
  //   ++instruction_num;
  // }

  //// james: below is to print out pruned algorithm, switch on to debug

  // for(std::vector<IntegerT>::iterator i = useful_list.begin(); i != useful_list.end(); ++i) {
  //   cout << "useful_list:" << *i << endl;
  //   // ++(*i);
  // }
  vector<double> figure_algo_pred;
  vector<double> figure_algo_learn; 
  // cout << "def predict:" << endl;
  IntegerT ins_countprint = 0;
  IntegerT char_num_count = 0;
  for (const std::shared_ptr<const Instruction>& instruction :
       algorithm.predict_) {
    // cout << "code run here 17 " << endl;
    // cout << "instruction->string: " <<instruction->ToString() << endl;
    // cout << "instruction->in1_" << memory_.scalar_[instruction->in1_] << endl;
    // cout << "instruction->out_" << memory_.scalar_[instruction->out_] << endl;

    if ((*useful_list)[algorithm.learn_.size() + ins_countprint] < 1) {
      ++ins_countprint;
      continue;
    } else {
      // cout << instruction->ToString() << endl;
      for (int i: instruction->ToString()) {
        if (char_num_count % 1 == 0) {
          figure_algo_pred.push_back((double)i);
          ++char_num_count;
          // cout << i << endl;
        }
      } 
    } 
   ++ins_countprint;
  }
  // cout << "def learn: " << endl;
  IntegerT learn_instr_num = 0;
  for (const std::shared_ptr<const Instruction>& instruction :
       algorithm.learn_) {
    if ((*useful_list)[learn_instr_num] < 1) {
      ++learn_instr_num;
      continue;
    } else {
      // cout << instruction->ToString() << endl;
      for (int i: instruction->ToString()) {
        if (char_num_count % 1 == 0) {
          figure_algo_learn.push_back((double)i);
          ++char_num_count;
          // cout << i << endl;
        }
      } 
    }
    // cout << instruction->ToString() << endl;
    // ExecuteMyInstruction(instruction, rand_gen_, &memory_, features);
    ++learn_instr_num;
  }

  IntegerT ins_counter = 0;
  for (const std::shared_ptr<const Instruction>& instruction :
       algorithm.predict_) {
    // ++ins_count;
    if ((*useful_list)[algorithm.learn_.size() + ins_counter] < 1) {
      ++ins_counter;
      continue;
    }

    if (instruction->op_ == 65 || instruction->op_ == 66 || instruction->op_ == 72 || instruction->op_ == 75) {
      if (instruction->op_ == 65 || instruction->op_ == 66 || instruction->op_ == 75) ++num_stock_rank_count;
      if (instruction->op_ == 72) ++num_TS_rank_count;
    } else if (instruction->op_ == 73) {         
      ++num_stock_rank_count;
      ++num_TS_rank_count;
    }
    ++ins_counter;
  }

  double sharpe_ratio;
  double average_holding_days;
  double strat_ret_vol;
  double annual_mean_strat_ret;
  double correlation_with_existing_alpha;

  vector<double> result_vector;
  result_vector.resize(14);
  // vector<vector<double>> preds_for_corr;
  const IntegerT num_of_stocks_to_approximate_rank = 1026;
  const IntegerT all_rounds = num_stock_rank_count + 1;

  vector<vector<vector<double>>> tasks_rank;
    // cout << "code run here 11 " << endl;
  vector<vector<double>> vec_for_push(num_of_stocks_to_approximate_rank, vector<double>(1220)); /// james: 1000 + 244 - 12 (train iterator minus 12) - 12 (valid/test iterator minus 12)
  for (IntegerT i = 0; i < (num_stock_rank_count + num_TS_rank_count); ++i) tasks_rank.push_back(vec_for_push);
  // cout << "code run here 12 " << endl;
  // QUESTION!!!!!! James: does this variables get reset to 0 every time this function gets called? Because the num of train tasks and select tasks are obviously different, does it affect the size of vector of vector? 
  // //james: get all_task_preds size
  // const unique_ptr<TaskInterface>& task_get_size = tasks_[0];
  // all_task_preds.reserve(tasks_.size() * task_get_size.ValidSteps());  
  // Use all the tasks.
  for (IntegerT i = 0; i < (num_stock_rank_count + 1) * tasks_.size(); ++i) {
    task_indexes.push_back(i);
  }

  if (functional_cache_ != nullptr) {
    // cout << " Is this part run? " << endl;
    functional_cache_bit_gen_owned_->seed(kFunctionalCacheRandomSeed);
    // cout << " Is this part run 2? " << endl;
    // functional_cache_executor.Execute(nullptr, nullptr, tasks_rank, this_round, task_index, num_stock_rank, num_TS_rank, num_of_stocks_to_approximate_rank, &train_errors, &valid_errors, useful_list);
    // num_train_steps_completed_ +=
    //     functional_cache_executor.GetNumTrainStepsCompleted();
    const size_t hash = functional_cache_->Hash(
        figure_algo_pred, figure_algo_learn, algorithm.predict_.size() + algorithm.learn_.size(), (num_stock_rank_count + 1) * tasks_.size());

    pair<pair<double, vector<double>>, bool> fitness_and_found = functional_cache_->Find(hash);
    if (fitness_and_found.second) {
      //       // write to file
      // std::ofstream outFile("/hdd1/james/google-research/automl_zero/debug_fec/killed_train_my_file_123.txt");

      // for (auto j = (train_errors).begin(); j != (train_errors).end(); ++j) {outFile << *j << " ";}
      // outFile << "\n";            

      // outFile.close();
      // std::ofstream outFileDiff("/hdd1/james/google-research/automl_zero/debug_fec/killed_train_my_file_diff_123.txt");

      // for (auto j = (valid_errors).begin(); j != (valid_errors).end(); ++j) {outFileDiff << *j << " ";}
      // outFileDiff << "\n";            

      // outFileDiff.close();
      // Cache hit.
      // cout << " killed? " << task.index_ << endl;
      // CHECK(task.index_ == 999);
      functional_cache_->UpdateOrDie(hash, fitness_and_found.first);
      // cout << "returned this test correlation: " << fitness_and_found.first.fisrt << endl;
      return fitness_and_found.first;
    } else {
      // // write to file
      // if (task.index_ == 0) {
      //   std::ofstream outFile("/hdd1/james/google-research/automl_zero/debug_fec/train_my_file_123.txt");

      //   for (auto j = (train_errors).begin(); j != (train_errors).end(); ++j) {outFile << *j << " ";}
      //   outFile << "\n";            

      //   outFile.close();
      //   std::ofstream outFileDiff("/hdd1/james/google-research/automl_zero/debug_fec/train_my_file_diff_123.txt");

      //   for (auto j = (valid_errors).begin(); j != (valid_errors).end(); ++j) {outFileDiff << *j << " ";}
      //   outFileDiff << "\n";            

      //   outFileDiff.close();        
      // }


      // IntegerT task_count = 0;
      // cout << (num_stock_rank_count + 1) << "(num_stock_rank_count + 1) " << endl;

      for (IntegerT task_index : task_indexes) {

        IntegerT this_round = (task_index / tasks_.size());

        task_index = task_index % tasks_.size();
        // cout << task_index << "task_index " << endl;
        // cout << this_round << "this_round " << endl;
        // task_index = rand_gen_->UniformInteger(0, tasks_.size());
        // if (task_count > 99) break;

        const unique_ptr<TaskInterface>& task = tasks_[task_index];
        CHECK_GE(task->MaxTrainExamples(), kMinNumTrainExamples);
        const IntegerT num_train_examples =
            train_budget_ == nullptr ?
            task->MaxTrainExamples() :
            train_budget_->TrainExamples(algorithm, task->MaxTrainExamples());
        double curr_fitness = -1.0;

        
        // if (task_index < num_of_stocks_to_approximate_rank) this_round = task_index; 
        // else this_round = rand_gen_->UniformInteger(0, num_of_stocks_to_approximate_rank);

        // cout << "this_round: " << this_round << endl;
        curr_fitness = Execute(*task, num_train_examples, algorithm, &all_task_preds, &all_price_diff, &tasks_rank, this_round, task_index, &num_stock_rank, &num_TS_rank, num_of_stocks_to_approximate_rank, all_rounds, useful_list);
          // cout << "code run here 10 " << endl;
        // task_fitnesses.push_back(curr_fitness);
        // all_task_preds.push_back(train_preds);
        // ++task_count;    
      }  

      //   // write to file
      // std::ofstream outFilee("/hdd1/james/google-research/automl_zero/train_my_file_123e.txt");
      //   for (auto i = all_task_preds.begin(); i != all_task_preds.end(); ++i) {
      //     if (!(i->empty())) {
      //       for (auto j = (*i).begin(); j != (*i).end(); ++j) {outFilee << *j << " ";}
      //       outFilee << "\n";            
      //   }
      // }
      // outFilee.close();
      // std::ofstream outFileDiffe("/hdd1/james/google-research/automl_zero/train_my_file_diff_123e.txt");
      //   for (auto i = cc.begin(); i != cc.end(); ++i) {
      //     if (!(*i).empty()) {
      //       for (auto j = (*i).begin(); j != (*i).end(); ++j) {outFileDiffe << *j << " ";}
      //       outFileDiffe << "\n";            
      //   }
      // }
      // outFileDiffe.close();

      // for (IntegerT i = 0; i < (num_stock_rank * all_rounds); ++i) {
      //   // Creating iterator to point to first 
      //   // element in the list 
      //   vector<vector<double>>::iterator itr = all_task_preds.begin(); 
      //   vector<vector<double>>::iterator itr_diff = all_price_diff.begin(); 
      //   // deleting the first element 
      //   all_task_preds.erase(itr); 
      //   all_price_diff.erase(itr_diff);    
      // }
      // cout << "num_TS_rank" << num_TS_rank << endl;
      // cout << "num_stock_rank" << num_stock_rank << endl;
      // cout << "all_task_preds.size()" << all_task_preds.size() << endl;
      // cout << "all_price_diff.size()" << all_price_diff.size() << endl;

      // / james: write to files
      // std::ofstream outFile("/hdd1/james/google-research/automl_zero/search_my_file_dde.txt");
      // for (auto i = all_task_preds.begin(); i != all_task_preds.end(); ++i) {
      //   // if ((*i).empty()) cout << "this row is empty preds!!!" << endl;
      //     if (!(*i).empty()) {
      //       for (auto j = (*i).begin(); j != (*i).end(); ++j) {outFile << *j << " ";}        
      //     }
      //   outFile << "\n";
      // }
      // outFile.close();
      // std::ofstream outFileEval("/hdd1/james/google-research/automl_zero/final_eval_my_file_dde.txt");
      // for (auto i = all_price_diff.begin(); i != all_price_diff.end(); ++i) {
      //   // if ((*i).empty()) cout << "this row is empty diff!!!" << endl;  
      //     if (!(*i).empty()) {
      //       for (auto j = (*i).begin(); j != (*i).end(); ++j) {outFileEval << *j << " ";}
      //     }      
      //   outFileEval << "\n";
      // }
      // outFileEval.close();


      // cout << "code run here 13 " << endl;
      // delete the previous timesteps missed for ranking
      for (auto i = all_task_preds.begin(); i != all_task_preds.end(); ++i) {
        // if ((*i).empty()) cout << "this row is empty preds!!!" << endl;
        if (!(*i).empty()) {
          for (IntegerT j = 0; j < (num_TS_rank * 13); ++j) (*i).erase((*i).begin());
        }
      } 
      // delete the previous timesteps missed for ranking
      for (auto i = all_price_diff.begin(); i != all_price_diff.end(); ++i) {
        // if ((*i).empty()) cout << "this row is empty diff!!!" << endl;
        if (!(*i).empty()) {
          for (IntegerT j = 0; j < (num_TS_rank * 13); ++j) (*i).erase((*i).begin());
        }
      } 

      double combined_fitness;
      //     CombineFitnesses(task_fitnesses, fitness_combination_mode_);

      // CHECK_GE(combined_fitness, kMinFitness);
      // CHECK_LE(combined_fitness, kMaxFitness);

      if (is_search > 0) {
        // check completeness and return IC to array
        if (CheckCompleteness(all_task_preds) && all_task_preds.size() != 0) {

          vector<vector<double>> valid_period_preds = GetValidTest(all_task_preds, true);
          vector<vector<double>> valid_period_diff = GetValidTest(all_price_diff, true);
          vector<vector<double>> test_period_preds = GetValidTest(all_task_preds, false);
          vector<vector<double>> test_period_diff = GetValidTest(all_price_diff, false);

          double correlation = Correlation(valid_period_preds, valid_period_diff);
          double max_dropdown = ComputeAllMeasure(valid_period_preds, valid_period_diff, &sharpe_ratio, &average_holding_days, &strat_ret_vol, &annual_mean_strat_ret, valid_strategy_ret, true);

          double existing_corre1;
          double existing_corre2;
          double existing_corre3;
          double existing_corre4;

          // my_alpha_101_allow_1000_low_corr_0_p100_t10_no_fec_1018791_new_afec panda11
          vector<double> existing_alpha_ret = {0.001224,-0.001608,-0.001391,-0.000672,0.000167,0.004311,0.002309,0.000706,0.002827,-0.000202,-0.001828,0.000779,0.001701,-0.002960,-0.002118,-0.000290,0.000997,0.001104,0.000500,0.000987,-0.000345,-0.001451,-0.000304,-0.000354,0.000171,-0.002191,0.006876,-0.000870,-0.001405,-0.000080,0.000818,0.001910,-0.002808,0.005351,0.007376,-0.007819,0.005116,-0.000451,0.000083,-0.006025,-0.001118,-0.000759,-0.000163,0.000819,-0.000431,-0.002831,0.002468,-0.006197,0.000165,0.003467,-0.001596,-0.000673,-0.006541,-0.000549,-0.005515,0.001711,-0.001739,0.006246,-0.001597,0.000007,-0.002548,0.001163,-0.006527,0.000296,0.002426,-0.001292,0.001962,-0.001304,-0.000724,0.001967,-0.000708,0.001031,-0.000681,-0.004833,0.000857,-0.002176,-0.002628,-0.002875,-0.004236,0.002247,-0.002525,-0.005086,-0.007315,-0.002520,0.003511,0.004163,0.003474,0.001585,0.003983,0.001598,-0.003088,-0.002285,0.000868,0.001198,0.002424,0.001169,-0.001485,0.000964,-0.003366,0.002195,-0.000218,-0.002492,0.003624,-0.004654,-0.003878,-0.002047,-0.000415,0.001502,0.004320,-0.000999,-0.003655,-0.000149,-0.001642,-0.008301,-0.005541,-0.002447};
          vector<double> existing_alpha_ret_test = {-0.001599,-0.000963,-0.004110,0.000290,-0.003303,0.006604,-0.000512,-0.004431,0.001947,-0.003298,0.001147,-0.003483,0.001902,-0.002309,0.002586,-0.004156,-0.001108,0.001460,0.002033,-0.005519,-0.003722,-0.003996,-0.001980,0.002942,-0.001990,0.006957,0.001399,-0.006340,-0.000551,-0.000345,0.002708,0.000006,-0.001902,0.001816,0.002528,0.001461,0.001595,0.001221,0.003618,-0.002809,-0.001231,-0.007098,-0.004042,-0.000532,0.003360,0.002274,-0.002574,-0.006822,-0.001745,-0.002829,-0.010261,-0.003414,-0.000613,-0.002542,-0.004989,0.002362,-0.003858,-0.001314,-0.001282,-0.001294,0.001792,-0.002720,-0.002812,-0.003088,-0.006543,0.003307,-0.002025,-0.004060,0.000821,-0.000701,-0.001232,0.000728,0.006457,-0.001944,0.002227,0.004321,0.003057,0.002156,-0.003656,-0.002231,0.000770,-0.004623,0.002438,-0.001151,0.001699,-0.001660,0.000810,-0.000366,-0.001277,-0.000355,0.004241,-0.000744,-0.005438,0.004717,0.000816,-0.009150,-0.006324,0.001075,0.002367,0.003602,-0.004215,-0.007240,-0.001721,0.001578,-0.001861,0.001590,0.003218,-0.002555,-0.006567,-0.001343,-0.000147,-0.003605,0.003136,0.002115,-0.002631,-0.000428};

          // neural_net_algorithm3_allow_1000_low_corr_1_seed_1333921_no_fec_all_tasks_new_afec panda13
          vector<double> existing_alpha_ret_2 = {-0.001883,-0.004164,-0.006156,-0.000723,-0.000825,-0.002015,-0.000632,-0.003243,-0.003910,0.003308,0.000792,0.002087,-0.003378,0.005663,-0.001174,0.001720,0.001232,0.000757,-0.003863,-0.003073,-0.000633,0.002073,-0.003278,0.000753,0.001118,0.001735,-0.004002,-0.001350,-0.000317,-0.002799,0.003504,-0.003172,0.000608,0.001685,-0.006826,-0.001381,-0.001753,0.004187,-0.004757,-0.004196,0.001598,-0.000740,0.000190,0.001127,0.002685,-0.005228,0.000215,-0.000128,0.000079,-0.002302,0.000538,-0.000879,-0.002178,0.003965,0.002216,0.000141,0.007763,0.003497,0.003843,-0.004704,0.002375,-0.005188,-0.001633,-0.003044,0.000029,0.004647,0.002163,0.001272,0.000245,-0.001880,-0.002365,-0.003260,-0.003719,-0.001339,-0.002481,0.000575,-0.001353,-0.003012,0.000328,0.003821,-0.002624,-0.001443,-0.000475,0.001466,0.000329,-0.000518,0.001251,-0.004228,-0.001035,0.002957,-0.000495,0.007179,-0.000867,0.000040,0.000502,0.000480,-0.004651,-0.004409,-0.002326,-0.006420,-0.000302,0.001331,-0.000605,0.001344,-0.001356,0.004333,0.000163,0.002953,-0.000440};
          vector<double> existing_alpha_ret_2_test = {0.001883,-0.000168,-0.001233,-0.008903,-0.000190,-0.002226,-0.001363,-0.002779,0.000221,-0.001752,-0.002399,0.001524,-0.003077,0.002494,0.000170,0.003929,0.000366,0.004866,0.002436,0.002544,-0.000586,-0.000198,-0.001823,0.003771,-0.000912,0.004233,-0.001814,-0.000543,-0.003658,0.002878,-0.000038,0.000492,0.001439,0.000853,-0.002887,0.000208,0.002399,-0.007973,0.000920,-0.002209,-0.003768,0.001465,-0.002088,0.000704,-0.005493,0.001465,0.002010,-0.000487,0.002145,-0.002397,0.002540,0.000213,-0.004127,0.002831,-0.000071,-0.001030,0.001294,0.002630,0.000194,-0.004437,0.000733,-0.000638,-0.001333,0.001072,0.001584,0.002991,0.001479,0.004084,-0.000618,0.000289,0.003202,0.002302,-0.008450,0.000985,-0.000001,-0.000397,-0.001335,-0.001445,0.005921,0.001771,0.000894,-0.010542,0.000522,0.001130,0.003289,0.000355,0.001103,-0.007805,-0.003135,-0.007460,0.000321,-0.002080,0.001830,0.001601,-0.001003,0.000425,-0.000107,0.000876,-0.003948,-0.003380,0.005800,0.004265,0.008672,-0.004443,0.001586,0.004149,-0.006370,0.005204,-0.005072,-0.005982};

          // my_alpha_prune_allow_1000_low_corr_2_seed_3977777_no_fec_all_tasks_p100_t10_for_afec panda10
          vector<double> existing_alpha_ret_3 = {0.000000,0.000996,0.005810,0.001938,0.004838,0.001349,0.004129,-0.002328,0.002332,0.001349,0.002789,0.000459,0.004275,0.005610,0.005839,0.010633,0.012400,-0.002561,0.001238,0.004556,0.004205,-0.000561,0.001207,0.002411,-0.001629,0.003842,-0.000276,0.007930,0.003751,0.001098,0.001995,0.007071,0.002794,0.001261,-0.002651,0.008355,-0.002822,-0.000389,0.002424,0.006228,0.000010,0.008488,0.004101,0.000661,0.007039,0.003213,0.005136,0.008039,0.007899,0.001179,0.000104,0.000346,0.003013,0.000127,0.000337,0.000693,0.001151,0.001002,0.004882,0.003844,0.001864,0.000555,-0.001499,-0.000278,-0.000455,0.001472,0.004381,0.003914,0.004423,0.003816,0.001980,0.005225,0.004845,0.000771,0.010243,0.005289,0.002162,0.007169,0.004747,0.005430,0.008685,0.003879,0.009971,0.007706,0.006059,0.006542,0.002298,0.002726,-0.003176,0.004193,0.004447,0.002970,0.001620,0.002125,0.000659,0.000749,0.000369,0.002472,0.002112,0.001695,0.000556,0.001992,0.002230,0.003090,0.004406,0.001973,0.001056,0.000098,0.000571,0.004778,0.006007,-0.000356,0.003619,0.002653,0.003842,0.000524}; 
          vector<double> existing_alpha_ret_3_test = {0.001250,0.002782,-0.000779,0.001649,0.000508,0.002558,0.000138,0.002681,0.004757,0.001635,0.002812,0.002991,0.002732,0.002794,0.006887,0.002794,0.002335,0.001737,0.000870,0.005134,0.004168,0.003142,0.003888,0.002514,0.003374,0.005096,0.008249,0.009558,0.002894,0.007235,0.006962,0.000113,0.000472,0.001790,0.000827,0.002128,-0.001719,0.002756,0.000963,0.003424,0.000215,0.003417,0.001978,0.007216,0.001786,0.004674,0.003526,0.004989,0.002272,0.003282,-0.000553,0.000558,0.005754,0.003677,0.003600,0.002926,0.003408,0.005909,0.002166,0.002074,0.003434,0.006721,0.003634,0.004644,0.009817,0.005543,0.001918,0.003861,0.001406,0.002334,0.000531,0.002973,-0.000970,-0.001851,0.002705,0.001924,0.001868,-0.000643,0.004210,0.002784,0.004263,0.003749,0.005839,0.004605,-0.001560,0.008045,0.008386,0.004860,0.002668,0.006251,0.002669,0.013193,0.003009,0.003011,0.007103,0.006148,0.003307,0.002379,0.003067,0.006108,0.001914,0.003999,0.004814,0.002551,0.003416,-0.001254,0.003909,0.005212,0.018763,-0.000817,0.008274,0.010592,0.000859,0.003324,0.001219,-0.000517};

          // my_alpha_101_allow_1000_low_corr_3_p100_t10_no_fec_1008994_new_rank panda10
          vector<double> existing_alpha_ret_4 = {0.0040756, 0.0031563, 0.000129051, 0.00373519, -0.000202312, 0.00158072, 0.00538197, 0.00100945, 0.00332539, 0.0024705, 0.00210515, 0.00295206, 0.002621, -0.000790793, -0.00168755, 0.00751675, 0.00288368, 0.00311604, 0.000808149, 0.00377772, -0.00352965, -0.00163943, -0.00455195, 0.00208286, -0.0034128, 0.00221667, 0.00245747, 0.00551712, 0.00368253, 0.00104424, 0.00118564, 0.00359743, -0.00273357, 0.00226265, 0.00603837, 0.00218229, 0.00289919, -0.00300532, 0.0029417, 0.00119422, 0.00062219, 0.000247094, 0.00411388, -0.000347996, 0.000822655, 0.000743444, -0.000333629, 0.00672859, 0.000540362, 0.0024341, 5.27486e-05, 0.00262916, 0.00499986, 0.000204389, -0.0027676, -0.000453751, -9.19366e-05, 0.00223643, 0.000648946, 0.00183285, -0.0010633, 0.00027096, 0.000891729, -0.000132679, 0.00329658, 0.00219543, -0.000815209, 0.00031807, -0.00380936, -0.000548591, -0.000909178, -0.00219513, 0.00101414, 0.00100196, 0.00239809, 0.00233066, 0.0011846, -0.000552008, 0.00217169, -0.00205833, 0.0025243, 0.00350743, 0.00151811, 5.77816e-05, 0.00293952, 0.000792696, 0.000330212, 0.00399359, 0.00134878, -0.00127587, 0.00186386, 0.00345011, 0.00215696, 0.00216855, -0.000666104, -0.000620272, 0.00248596, 0.00236709, 0.00056133, 0.0025395, 0.000756568, 0.00383931, -0.00126461, 0.00139459, -0.00127253, 0.00107539, 0.00159618, -0.00149992, 0.00010252, 0.0010096, 0.000521808, 0.00230177, 0.000694147, -0.000348349, -0.0022005, -0.000207427};
          vector<double> existing_alpha_ret_4_test = {0.00119999, 0.000301742, 0.00302722, 0.00173234, 0.00111756, -0.00218288, -0.00101641, -0.000318455, 0.0024683, 0.00661973, 0.000850705, 0.0010818, -0.00235552, 0.0027329, -0.0021186, 0.00014327, 0.00468092, 0.000622064, 0.00183986, 0.0011201, 0.000779347, 0.00179028, 0.0065174, 0.00107612, 0.000783914, -0.00430983, 0.00182168, 0.000998737, 0.00162662, 0.000603166, -0.000575616, 0.00130931, -0.000291927, -0.00044126, 0.00335834, -0.00236152, -3.48376e-05, 0.00165223, -0.00050383, -0.00120481, 0.00346838, 0.00151728, 0.00228353, 0.000352406, 0.00211582, -0.000921195, -0.0010169, -0.000187047, 0.00548542, 0.00108787, 0.0124939, 0.00214241, -0.000187598, -2.5773e-05, 0.00510192, 0.003419, 0.000525343, 0.00141914, -0.000610481, -0.00058619, -0.00244232, -0.000129933, 0.00309982, 0.000550199, -0.00137139, -0.000291977, -0.00234732, -0.000692767, 0.00292385, 0.0027203, 0.000373473, -0.000782622, -0.000250607, 0.000295233, 0.000224251, 0.000489861, -0.00317021, -0.00164023, 0.000833466, 0.00236122, -8.35385e-05, 0.00208497, 3.83559e-05, 0.000550012, -0.00454016, 0.000271052, -0.00170992, 0.00303823, -0.0024213, 0.00168468, 0.00100232, 0.00384892, 0.00326069, -0.00235716, 0.000161643, 0.00911055, 0.000332738, -0.000423506, -0.00178251, -0.000635346, -0.00215373, 0.00562947, 0.00171501, -0.00949365, 0.00268461, -0.00253421, 0.00199781, 0.00129538, 0.00290936, 0.00239678, 0.0015036, 0.00709739, -0.00208643, -0.000887032, 0.000533674, 0.00168763};

          // // best corre0 my_alpha_prune_allow_1000_low_corr_0_seed_2221999_no_fec_all_tasks_p100_t10_for_afec_exp0 panda13
          // vector<double> existing_alpha_ret = {0.00312073, -0.00161748, 0.00353822, 0.00246916, 0.00174183, 0.000849173, 0.000429936, -0.00116662, 0.000393727, 0.000275426, 0.00644967, -0.000990445, 0.00574846, 0.00636272, 0.00797232, 0.00843352, 0.00879795, -0.0038535, -0.00023114, 0.00192809, 0.00678282, 0.00356124, 0.00214842, 0.00188346, -0.00287907, 0.00565348, 0.00142204, 0.00707139, 0.00274133, 0.00250949, 0.00261631, 0.00628416, 0.00534709, 0.00201456, -0.00406821, 0.00775719, -0.00243215, -0.000182053, 0.0023179, 0.00635213, 0.00178486, 0.00717796, 0.00362536, 4.29516e-05, 0.00789155, 0.00270866, 0.00514831, 0.0100887, 0.0107886, 0.00187376, -0.000382351, 0.000288335, 0.00244406, -0.00127711, 0.00549015, -0.00453049, 0.00178179, 0.00245515, 0.00460936, 0.00243234, 0.00149055, 0.000710177, 0.00187084, -0.00214315, 0.000655005, 0.00248573, 0.00459578, 0.00161737, 0.0044974, 0.00625413, 0.00239157, 0.000193567, 0.0026638, 0.00129554, 0.0100933, 0.00880633, 0.00191173, 0.00736741, 0.00611324, 0.0037481, 0.00756467, 0.00505861, 0.0104252, 0.00776401, 0.00544443, 0.00581822, 0.00549653, 0.00372955, 0.000790406, 0.00319583, 0.00372207, 0.003395, 0.00306553, 0.00143447, 0.00161238, 0.00029676, -0.00103598, 0.000691722, 0.00340233, 0.00388392, 0.000539224, 0.00424223, 0.00207928, 0.00395826, 0.00281807, 0.00402886, 0.00154108, 0.000365914, 0.000117177, 0.00405567, 0.00633325, 0.00161281, 0.00468035, 0.00332887, 0.00456248, 0.00200756};
          // vector<double> existing_alpha_ret_test = {0.00299815, 0.00308175, -0.00106391, 0.00153861, 0.000525495, 0.00212095, -0.000369674, 0.00259607, 0.00309297, 0.0012399, 0.00242144, 0.0026552, 0.0040542, 0.0038409, 0.0059543, 0.00133803, 0.0027604, 0.00414744, 0.0041918, 0.00402953, 0.00617217, 0.00377027, 0.00416956, 0.00258698, 0.00523881, 0.00802828, 0.0117286, 0.00965956, 0.00459794, 0.00784191, 0.0065103, 3.42815e-05, -0.000325954, 0.00365467, 0.00108213, -0.000178369, 0.00108331, 0.000963741, 0.00104766, 0.00444664, 0.00103942, 0.00442437, 0.00346027, 0.00596707, 0.00218263, 0.00512517, 0.00245906, 0.00548905, 0.00353745, 0.00204963, -0.000703199, -0.000678646, 0.00516429, 0.00335553, 0.00461882, 0.00328339, 0.00144221, 0.00459146, 0.00317071, 0.00708109, 0.0032176, 0.00592698, 0.00276652, 0.00487139, 0.00628714, 0.00709793, 0.0034373, 0.00276068, 0.00148488, 0.00292563, 0.00170982, 0.00292891, 0.00107186, -0.000331041, 0.0033626, 0.00271712, 0.000395023, -0.000587284, 0.00509573, 0.00341463, 0.00424587, 0.0073924, 0.00635192, 0.0065039, 0.000426408, 0.00533888, 0.00744941, 0.00960461, 0.00315154, 0.00904076, 0.00588764, 0.011615, 0.00591185, 0.00172891, 0.00370427, 0.0105161, 0.00467994, 0.00190401, 0.00291213, 0.0067943, 0.00372354, 0.00479721, 0.00313099, 5.09552e-05, 0.00434685, 0.000540275, 0.00371784, 0.00300241, 0.0118237, 0.0078223, 0.00545857, 0.00266656, 0.00616071, 0.000684343, 0.00479074, -0.00254374};

          // // genetic_alpha_own_pred
          // vector<double> existing_alpha_ret = {-0.002686928483661588, -0.002129339156824561, 0.007878488275612439, -0.0001403472117990079, 0.004986423061670964, -0.0032943385922352686, 0.00013572078460866166, -0.0013795144540375004, 0.00039669857193525004, 0.00043402158136252034, 0.003185497042567098, -0.001824264384622798, 0.0010294074298979883, 0.005744926320088828, 0.0009092506382315513, 0.0019443188946131063, 0.0017380194781331237, -0.006063039574963791, -0.0005534018775105176, -0.0012743247474090724, 0.005563157936984364, -0.0010362878970668898, 0.0043741250565108025, -0.0003124555515507943, -0.003440449069501139, -0.0013517576260835273, -0.008782162748910638, 0.00430779094112177, 0.002301362438214216, -0.0013730631556445605, -7.455064879913209e-05, 0.004211635997887875, 0.0026440995239680465, -0.005424585038549923, -0.008366532884780953, 0.011453369543746872, -0.004571637393341765, 0.0006162052726381706, -0.00010624509584389319, 0.00673265107050347, -0.0019603468755066134, 0.00090572232831021, 0.0016956176320641614, -0.0011960528860186503, 0.0010493498060979434, 0.004274235952066885, -0.0015595804597801077, -0.0010613203791470793, 0.006886781410185616, -0.004170786153691153, 0.0016725261092267463, 0.0019885691109990944, 0.00012792402441097472, 0.0018257558999861256, 0.007051985153614382, -0.0027318515905525587, 0.00048506416962057486, -0.004456230412701823, 0.002216575234599727, 0.00012774155729555758, 0.0017028785598924845, -0.0005637737983773539, 0.004649564895415681, -0.0022863372131446402, 0.002979783079814391, 0.0011264681318448044, -0.004661232176752073, 0.0010768245441148405, -0.003137513343609344, -0.004240438726074602, 0.005225347279070647, 0.0006251327539299467, 0.0005283697524534059, 0.0006014210052005531, -0.001440413333230861, 0.00862724106142343, 0.002201312150166501, 0.0007156443790472533, -0.0013114234867942498, -0.0013230530270398333, 0.005060556286689222, 0.0013460449968725197, 0.0034200974120206507, 0.0020025895114978987, -0.0011618456265115595, 0.000999216393565483, -0.0028547366034301636, -0.0008184197200991061, 0.001517328466112744, -0.0002168271808711264, 0.004464960864322798, 0.0008135361466954372, -0.0026101742761627245, -0.0010591795522213454, -0.003072593626893827, -0.002650880653889298, -0.0058526984911870805, -0.00567888953963247, 0.0020288847841343216, -0.004567863815021389, -0.004199919087500281, 0.0005639539559902929, -0.0030990551192039417, -0.000132729618477323, 0.016661055544755943, 0.006365392178922624, -0.0002866746129477349, -0.004850991525145898, -0.0037609348833247402, -0.000558700193814432, 2.4675143396768462e-05, 0.0019721836567878626, 0.0012760436935754793, 0.006430752331335832, 0.003827501428431024, 0.006714069974997683};
          // vector<double> existing_alpha_ret_test = {0.003282821371569211,0.002631987310274253,0.000203727406409282,0.0004887027095044072,0.0026374272929656772,-0.0027713213448975482,0.0033177731728812887,0.0010049371320306477,-0.0008854599006399289,0.0025053858123165185,0.0028037666442637388,2.4243270879020784e-05,-0.0016805265682210413,0.003044911890798252,0.007871837393534253,0.0057751230612839155,0.003166609705317125,7.984494655643992e-05,-0.00019454060755674796,0.0025001009016085707,0.0022672062645756608,-0.003751268201277469,0.0053924316924296445,0.0008945115812977189,0.004381392888294533,0.004588066795548906,0.0036952403645214016,-0.0015490041989849601,0.0028815725041744233,-0.0010942433960563491,0.0002647604927743519,0.0072330801262263655,0.003882948889329274,0.0004748537873942027,0.002790704380850606,6.576401918612085e-05,-0.0002947682586348366,0.0031168886356165437,0.0037957665991217304,0.0021982811532921254,0.006258027890148954,0.004614381494193465,0.0024869742900759384,0.0007591682556000734,0.0028629877906445333,0.006188825166939083,-8.579499702543458e-05,0.0019879980858872326,0.0013614359013105481,-0.0030237303915043867,0.0008445549469349167,0.0026824577672721617,0.0037876306030175666,0.00027310157874271823,0.001138808233319022,8.510667272121353e-05,0.0017617409922519034,0.00048399533682608187,-0.00019917478390996113,0.000942884412379108,0.0017992182212276386,0.001628077077131751,0.0004929013275807304,-0.0006259501641123766,0.003468563436797245,0.008411860957481698,0.001294003882133632,0.0010589684706541203,0.0015607193188043045,0.0058456116190537255,0.000841531199664125,-0.0028362273236779423,-0.0026981036259308144,0.005089121324283186,0.000134321304153584,0.007393784041658691,0.0036980041953544873,0.006776007355567826,-0.0004300285261208403,0.0039338834193101135,0.002495234986194639,-0.0005838718679945787,0.004000888690571136,0.006267031210937857,0.010271104184777613,0.002058202336292947,0.008249482724035362,0.005439766294035309,0.0055704354117764865,0.0030840066412227696,0.0014823719385257395,0.006954010814504086,0.0030870132373586046,0.001438368726878858,0.0012298764742872947,0.00020471189303106208,-0.0017123504205638351,0.003283085821048637,0.0017548564932672317,0.0069737076507718765,0.003752395039692047,0.003585348664291388,0.0014967745960550172,0.0027841341939094377,-0.0015651642062547433,0.0008264051353374935,0.00012586283826143685,0.00149528944266053,0.002021653535087875,0.0007276194043681627,0.0018936728196023989,-0.00028920948896515863,-0.0003941593422175371,0.003187178376683697,-0.0008779504019585938,0.006054446072416786};

          // // alpha_101
          // vector<double> existing_alpha_ret = {0.00303132, -0.002403, -0.000904871, 0.00060278, 0.00125142, 0.00398978, 0.000626033, 0.00154211, -0.00263525, 0.0029778, -0.00616227, 0.00239627, -1.41221e-05, 0.00017999, 0.000321472, 4.59936e-05, 0.0043534, 0.000942985, -0.000799042, -0.0004227, -0.000936619, 0.00221591, -0.000792551, 0.00187288, -0.00130669, 0.0017982, 0.00221199, -0.00151729, -0.00353836, -0.00123755, -0.00245238, 0.00310715, -0.00435041, 0.00217772, 0.00438846, 0.00577187, 0.00382205, 0.000356219, 0.000167673, -0.00431133, -0.000989082, -0.00135114, -0.000154142, -0.00143834, 0.00333935, 0.00513699, 0.00204901, 0.0036684, -0.0114912, -0.00115769, 0.00494539, 0.00264235, -0.000841094, -0.00012577, 0.00205934, 0.00680964, 0.00318218, 0.00491436, 0.000300511, 0.00109713, 0.000328575, 0.0017116, -2.22631e-05, 0.00254269, 0.00417653, -0.00199953, 0.00371213, 0.00145323, 0.00027628, -0.00344761, 0.00107935, 0.00356198, 0.00169508, 0.00267984, 0.000952758, -0.00413732, 3.43873e-05, 0.00193003, -0.00567967, -0.000624276, 0.00274116, 0.000295068, -0.00415611, -0.00407138, -8.74978e-05, -0.00266056, 0.000880871, -0.000917506, 0.00518949, -0.000759792, 0.00654163, 0.00147564, -0.000457407, 0.000411151, 0.00120337, 0.00066914, -0.000559559, 0.00100896, 0.00479035, -0.00534908, 0.00210131, 0.00184155, 0.00398438, -0.00136632, -0.00466098, 0.000858508, 0.0031966, -0.00141343, -0.00342208, 7.36149e-05, 0.00362966, -0.000477499, -0.00391243, -0.00181578, -0.00136947, 0.00342638};
          // vector<double> existing_alpha_ret_test = {-0.00266955, 0.000450922, 0.00124023, -0.00181689, 0.00217626, 0.00944221, 0.00177761, -0.000565231, 0.00147649, -0.00341887, -0.000904172, 0.000165574, 0.00356002, 0.000535788, 0.000580993, 0.000818235, 0.000225145, -0.00431209, -0.000943655, 0.0070442, 0.00326737, 0.00263834, 0.00454103, -0.00263583, 0.00170998, 0.00322186, -0.000301089, 0.000342297, 0.00306343, 0.000219519, 0.00637094, -0.00149522, -7.53497e-05, 0.00527052, 0.00321139, 6.01136e-05, -0.000651166, 0.000989465, 0.00226964, 0.00150764, -0.000138668, -0.00245506, 0.00247056, 0.00178129, -0.00174677, -0.00266464, -0.000197332, 0.00276214, 0.000132817, 0.00202388, -0.00188228, 0.00352316, 0.000233494, -0.00145179, 0.000203296, 0.00150056, 0.00286276, -0.000101587, -0.00540683, -0.00182192, -0.000795945, -0.000949182, 0.000324852, -0.0021186, -0.00101324, -0.00141126, 0.00176375, 0.00416738, 0.00106224, 0.000360688, 0.0041462, -0.00125153, -0.000291151, 0.00221045, 0.000473127, -0.00400749, 0.00250879, -0.00194112, 0.00427002, 0.000667661, -0.00233414, 0.00039995, 0.000442779, 0.00325223, 0.00083162, 0.00155445, -0.000863141, 0.000568708, 0.0022164, 0.00254289, 0.00566946, 0.00589683, 0.00148539, -0.00155038, -0.00213979, 0.00508064, -0.000588039, 0.0013566, 0.000359694, 0.00210272, 0.00135259, 0.00242127, 0.000849111, -0.000501287, 0.00457756, 0.00293086, 0.00657017, -0.00376364, -0.0129054, 0.00490065, -0.00444022, 0.00165547, 0.00458949, -0.00144161, -0.00160465, -0.000632827};

          // // neural_net_algorithm_allow_1000_low_corr/neural_net_algorithm_recordperf15400.txt
          // vector<double> existing_alpha_ret = {0.006197,0.008935,0.001561,0.004624,-0.001808,0.003319,0.000075,0.003401,0.000474,0.003744,0.001253,-0.000175,0.001856,-0.004973,0.004354,0.000520,0.000011,0.003512,0.001230,0.006569,-0.003931,-0.009860,0.013157,-0.006577,0.000203,0.000618,0.008907,-0.000961,0.005416,0.000240,-0.001292,0.005166,0.004745,0.000760,0.000655,0.007175,-0.002170,-0.000104,0.000121,0.004989,0.002100,0.005266,-0.003300,0.001625,-0.002970,0.002356,0.002939,0.003586,0.001805,0.007846,-0.002752,0.001296,0.002708,-0.000918,0.002025,0.000015,0.002918,0.004034,0.000405,0.002664,0.001895,0.003335,0.007175,0.001234,0.006015,0.008324,0.002481,0.005788,0.003903,0.005935,0.003954,-0.001614,0.000662,-0.002567,0.002976,0.000895,-0.000531,0.007255,0.004105,-0.001800,-0.001302,-0.002926,-0.000566,0.001451,-0.004039,0.004261,-0.004525,-0.001501,0.000675,-0.005413,0.000373,0.013984,0.005851,0.000064,-0.001860,-0.001959,0.003308,0.003082,0.002760,0.006530,0.009937,0.007792,0.003174,0.006275,0.000104,0.007505,0.001078,0.005330,-0.006551};
          // // neural_net_algorithm_allow_1000_low_corr/neural_net_algorithm_recordperf15400.txt
          // vector<double> existing_alpha_ret_test = {0.002478,0.002715,0.001446,0.006236,-0.000316,0.005457,0.002920,0.001683,-0.000255,0.008116,0.004947,-0.000104,-0.003969,0.010330,0.002454,0.007914,0.001814,0.001464,0.000749,-0.003763,0.001426,0.015447,-0.001082,0.004240,-0.003034,-0.000190,-0.000768,-0.002212,-0.003602,-0.000189,0.002330,-0.001858,-0.004480,0.005809,0.004281,0.009493,0.003460,-0.001592,0.000611,-0.001230,-0.000932,0.007642,0.001132,0.006371,0.004827,0.003985,0.002757,0.006976,0.010436,0.003048,0.004871,-0.000761,-0.002923,0.003635,0.000708,0.001965,0.007537,0.003807,0.004422,-0.000385,0.002027,0.002957,0.000836,0.003972,0.000322,0.000832,-0.007178,0.000824,0.000231,-0.002217,-0.002451,-0.004861,0.007245,0.001596,0.002583,0.004047,-0.000102,-0.000512,-0.001053,0.002895,-0.000775,0.007445,-0.002357,0.002756,-0.005853,0.002086,0.000527,-0.005383,0.001275,0.013588,0.001596,-0.000193,-0.001583,0.006656,0.000785,0.010209,0.003550,-0.003665,0.006402,-0.005552,0.006601,0.007244,0.021032,-0.002978,0.005590,0.001256,0.000155,-0.002903,-0.000166,0.001548};
         
          // // Second alpha 1 for low correlation experiment\my_alpha_101_allow_1000_low_corr_seed_1000080\my_alpha_101_recordperf8300
          // vector<double> existing_alpha_ret_2_test = {-0.000076,-0.003797,-0.001911,0.002350,0.004417,0.005409,0.004090,-0.001692,0.003475,0.004007,-0.000260,0.003324,0.001721,0.002167,0.003281,0.001562,0.000735,0.001208,-0.001045,0.003420,0.006600,0.005770,0.001991,0.000170,0.001028,0.004738,-0.000064,0.002194,0.002637,0.002609,0.004757,0.004471,0.000132,0.002744,0.001939,0.002918,-0.002643,-0.001062,-0.001843,0.001878,0.003933,-0.000721,-0.000692,0.001038,-0.001743,-0.007462,0.001993,0.003188,0.006034,0.001352,0.003035,0.002305,-0.001755,0.000140,0.002028,0.000334,-0.001049,-0.002329,-0.000813,-0.000437,0.000081,-0.001449,0.003242,-0.001082,-0.009997,0.002829,0.003748,0.001607,0.001326,-0.000567,0.002552,0.001623,0.000117,0.000205,0.000290,0.000628,0.001375,0.000268,0.001854,0.000797,-0.001198,-0.001318,0.000295,0.001351,-0.001728,0.002518,0.000986,0.005011,0.002962,0.012786,0.005997,0.007253,0.005815,-0.003374,0.003251,0.005252,0.000900,0.003219,-0.001560,0.003883,0.000452,0.004338,0.000211,0.000535,0.007252,0.001341,0.007201,-0.001422,-0.002765,0.009425,0.000206,0.000263,0.005351,-0.000008,0.002212,-0.003677};
          // vector<double> existing_alpha_ret_2 = {0.003445,-0.002405,0.003635,0.001086,0.011992,0.001228,0.003432,-0.000996,-0.000907,-0.000562,-0.000600,0.000726,0.004271,0.005852,0.001955,0.004330,0.006150,-0.000057,0.001039,0.001596,0.003611,-0.001835,-0.002375,0.002001,-0.002022,0.000678,0.002831,0.003616,0.001183,0.002565,-0.003314,0.002114,0.000079,0.005419,-0.003359,0.012499,0.002233,0.000154,0.000352,0.000943,0.000010,0.005770,0.001124,-0.001031,0.003592,0.004287,-0.001482,0.007629,0.000244,-0.000099,0.003255,-0.000412,-0.000078,-0.001176,0.003383,0.003023,0.004283,0.001050,0.000708,0.003390,0.000427,-0.000727,0.002275,0.000759,0.001818,-0.000193,0.003332,0.001092,0.002366,-0.002524,0.002272,-0.000525,0.001394,0.000475,0.005509,0.003184,-0.000333,0.004161,0.006384,0.002879,0.006985,0.003377,-0.004621,0.001884,0.002198,0.002126,-0.000626,-0.000944,0.003638,-0.004099,0.002882,0.002149,0.001971,0.001933,0.001419,0.000484,0.000677,-0.002105,0.002859,-0.005282,0.002227,0.000701,-0.000035,0.002998,-0.012669,0.000772,-0.000243,-0.000462,-0.000128,0.005970,0.006057,-0.000881,-0.004189,-0.001998,0.001194,0.005108};

          // // Third alpha 2 for low correlation experiment\neural_net_algorithm_allow_1000_low_corr_2_seed_1000150_no_fec_all_tasks_p100_t10\neural_net_algorithm_recordperf10000
          // vector<double> existing_alpha_ret_3_test = {-0.004826,0.006961,-0.005362,-0.000832,-0.001797,0.003694,-0.001292,0.009858,0.003371,0.002377,0.006323,-0.002485,0.002797,0.001930,0.004070,0.004766,0.002207,0.000898,0.009377,0.001260,0.000104,-0.007495,0.002988,-0.003090,0.002792,-0.003235,0.002360,0.006547,0.007068,0.001724,0.001499,-0.006402,0.001805,0.006385,0.002870,-0.003985,-0.002815,0.000879,0.001231,0.002175,-0.003547,-0.001037,-0.003550,0.007964,-0.000413,0.006120,0.003590,0.000344,0.000029,0.001594,0.012429,0.001667,0.005441,-0.001795,-0.001609,0.000787,-0.002739,0.005928,0.005993,-0.002427,0.000929,0.002673,-0.008503,0.001477,0.013194,-0.001440,0.000690,0.002454,-0.000745,0.002495,-0.003148,-0.001721,-0.000386,-0.000695,0.000851,0.000644,-0.004362,0.000325,0.001637,0.004203,-0.001367,0.004598,0.008026,0.001535,-0.002760,0.004657,0.004796,0.002224,0.005338,-0.005469,0.000931,0.006911,-0.000270,-0.000144,0.005148,0.000271,0.004605,0.005494,-0.000398,0.001798,0.000440,-0.003256,0.002434,0.003085,-0.002748,0.002986,-0.001495,0.000817,-0.006244,0.003631,0.000496,-0.007383,0.007119,0.001236,0.008052,0.000660};
          // vector<double> existing_alpha_ret_3 = {-0.001147,0.001377,0.000493,0.004362,-0.005155,-0.000165,0.002030,-0.000922,0.001502,0.003023,0.003925,-0.000381,0.003788,-0.000735,0.001789,0.011604,0.002698,0.004129,-0.003751,0.004330,0.001844,0.001878,0.000799,0.002163,0.002871,0.003333,-0.000974,0.007340,0.004750,-0.000780,0.004697,0.002937,0.004184,-0.001463,-0.001772,0.002209,0.001133,-0.000254,0.003978,0.005899,0.004853,-0.000103,0.005980,-0.000252,-0.000154,0.004933,-0.000341,0.008214,-0.001518,0.000722,-0.000480,0.006710,0.001693,0.000440,-0.002558,0.003312,0.000622,0.000015,0.006928,0.001842,0.003243,-0.002033,0.003488,-0.001547,0.000168,0.000633,0.003189,0.001664,0.004608,0.003879,0.000103,0.005875,0.004149,0.002224,0.012888,-0.002514,0.009278,0.002516,0.001498,0.004237,0.008121,0.004060,0.007216,0.006147,0.002899,0.003551,0.003666,0.003494,-0.010394,0.007991,0.005138,0.004506,-0.000289,0.002877,0.003927,0.000183,0.000538,0.002418,0.003377,0.004013,-0.001013,0.002590,0.001944,0.004537,-0.013242,-0.003105,0.003767,-0.000557,-0.000396,0.003609,0.009688,-0.000151,0.006744,0.000370,-0.002976,-0.000649};

          // // Fourth alpha 3 for low correlation experiment\my_alpha_101_allow_1000_low_corr_3_seed_1000090_no_fec_all_tasks_p100_t10_original
          // vector<double> existing_alpha_ret_4_test = {0.003984,-0.000642,0.000168,0.000743,-0.000759,0.001906,0.000159,-0.002147,-0.002990,-0.001171,-0.001303,0.001935,0.000345,-0.002126,-0.000226,-0.002363,0.000445,-0.000674,-0.001109,0.001907,-0.001557,-0.002643,-0.000666,0.001333,-0.000590,0.003530,0.001440,0.003135,0.001000,-0.000760,-0.002173,-0.001678,-0.000129,-0.003787,-0.001139,0.001048,0.002541,-0.000339,0.000264,-0.001413,0.000084,0.001861,-0.002316,-0.000542,0.001433,-0.000140,-0.001479,-0.004338,-0.002398,-0.002228,-0.008188,-0.001347,0.002009,0.001959,0.002559,0.001556,0.000020,0.000089,-0.001116,0.002859,-0.000839,-0.000025,-0.001862,0.002123,-0.006681,-0.000361,0.000139,-0.002231,-0.000006,-0.001144,0.003094,-0.000638,-0.001649,0.001697,0.002686,0.001057,0.003905,0.000388,-0.001070,-0.000535,0.001093,-0.000861,0.002083,-0.004180,-0.003213,0.001763,-0.002662,0.001582,-0.001573,0.000371,-0.000203,-0.000451,0.002239,0.001635,0.001498,-0.009242,-0.001236,-0.000145,0.005236,-0.001278,-0.001869,0.001770,0.001549,0.000595,0.003334,-0.001451,0.001385,-0.001763,0.000643,0.000894,0.002622,0.000844,0.000733,0.001693,0.000134,-0.002630};
          // vector<double> existing_alpha_ret_4 = {0.001984,-0.000233,0.001332,-0.000506,0.001348,-0.000378,0.001397,-0.002566,0.000224,-0.002151,-0.004171,-0.001391,0.000482,0.002737,-0.002964,-0.002752,0.001146,-0.001926,0.003205,0.000985,0.000036,0.001523,-0.000224,-0.001516,-0.002211,0.001650,-0.001049,-0.000215,-0.001640,0.000963,0.001383,-0.000929,0.000398,-0.000068,-0.002211,-0.000525,0.001870,0.002027,-0.001525,0.000695,0.000636,-0.001390,0.000476,0.000489,-0.003071,-0.001675,-0.002035,-0.001344,0.003025,-0.000678,-0.001944,-0.000871,0.001351,-0.001206,-0.000617,0.000910,-0.000050,0.001452,-0.005430,0.002297,0.000412,-0.001499,-0.000999,0.000180,-0.000804,-0.000564,-0.001047,-0.001135,-0.000859,0.000543,0.001258,-0.000293,0.001728,-0.004996,0.002041,-0.001702,0.000848,0.002812,-0.002644,0.000527,-0.000864,-0.000434,-0.000571,0.000507,0.001994,0.001688,0.001052,-0.002481,-0.000227,0.000358,-0.000866,0.000011,-0.000894,0.002424,0.000979,0.001194,-0.004281,-0.003781,-0.002728,-0.001790,-0.000757,-0.001055,-0.001342,0.003044,-0.001710,0.001409,0.002240,0.000179,0.002029,0.002146,-0.002095,0.001466,-0.002633,0.002332,0.002048,0.001100};
     
          existing_corre1 = CorrelationWithExisting(valid_strategy_ret, existing_alpha_ret, all_task_preds[0].size(), existing_alpha_ret_test.size());
          existing_corre2 = CorrelationWithExisting(valid_strategy_ret, existing_alpha_ret_2, all_task_preds[0].size(), existing_alpha_ret_2_test.size());
          existing_corre3 = CorrelationWithExisting(valid_strategy_ret, existing_alpha_ret_3, all_task_preds[0].size(), existing_alpha_ret_3_test.size());
          existing_corre4 = CorrelationWithExisting(valid_strategy_ret, existing_alpha_ret_4, all_task_preds[0].size(), existing_alpha_ret_4_test.size());

          correlation_with_existing_alpha = (existing_corre1 + existing_corre2 + existing_corre3 + existing_corre4) / 4;// 
          
          
          if (existing_corre1 > 0.15 || existing_corre2 > 0.15 || existing_corre3 > 0.15 || existing_corre4 > 0.15) correlation = 0;// 
          result_vector[0] = correlation;
          result_vector[1] = sharpe_ratio;
          result_vector[2] = average_holding_days;
          result_vector[3] = 1 - max_dropdown;
          result_vector[4] = strat_ret_vol;
          result_vector[5] = annual_mean_strat_ret;
          result_vector[6] = correlation_with_existing_alpha;
          // ICs.push_back(correlation);
          // cout << "valid sharpe_ratio: " << sharpe_ratio << "; valid correlation: " << correlation << "; valid max_dropdown: " << 1 - max_dropdown << " correlation_with_existing_alpha: " << correlation_with_existing_alpha << std::endl;
          // combined_fitness = 100 * correlation + sharpe_ratio + 0.1 * average_holding_days + max_dropdown;
          combined_fitness = correlation;

          // test part
          double test_correlation = Correlation(test_period_preds, test_period_diff);
          double test_max_dropdown = ComputeAllMeasure(test_period_preds, test_period_diff, &sharpe_ratio, &average_holding_days, &strat_ret_vol, &annual_mean_strat_ret, strategy_ret, false);

          double existing_corre1_test = CorrelationVec(existing_alpha_ret_test, *strategy_ret); 
          double existing_corre2_test = CorrelationVec(existing_alpha_ret_2_test, *strategy_ret); 
          double existing_corre3_test = CorrelationVec(existing_alpha_ret_3_test, *strategy_ret); 
          double existing_corre4_test = CorrelationVec(existing_alpha_ret_4_test, *strategy_ret); 

          correlation_with_existing_alpha = (existing_corre1_test + existing_corre2_test + existing_corre3_test + existing_corre4_test) / 4;//  

          // if (existing_corre1_test > 0.15 || existing_corre2_test > 0.15 || existing_corre3_test > 0.15) test_correlation = 0;
          cout << "test sharpe_ratio: " << sharpe_ratio << "; test correlation: " << test_correlation << "; test max_dropdown: " << 1 - test_max_dropdown << " correlation_with_existing_alpha: " << correlation_with_existing_alpha << std::endl;
          result_vector[7] = test_correlation;
          result_vector[8] = sharpe_ratio;
          result_vector[9] = average_holding_days;
          result_vector[10] = 1 - test_max_dropdown;
          result_vector[11] = strat_ret_vol;
          result_vector[12] = annual_mean_strat_ret;
          result_vector[13] = correlation_with_existing_alpha;
          // cout << "size: " << check_cycle_list->size() << endl;
          // // write to file
          // std::ofstream outFile("/hdd1/james/google-research/automl_zero/train_my_file_123.txt");
          //   for (auto i = test_period_preds.begin(); i != test_period_preds.end(); ++i) {
          //     if (!i->empty()) {

          //       for (auto j = (*i).begin(); j != (*i).end(); ++j) {outFile << *j << " ";}
          //       outFile << "\n";            
          //   }
          // }
          // outFile.close();
          // std::ofstream outFileDiff("/hdd1/james/google-research/automl_zero/train_my_file_diff_123.txt");
          //   for (auto i = test_period_diff.begin(); i != test_period_diff.end(); ++i) {
          //     if (!i->empty()) {

          //       for (auto j = (*i).begin(); j != (*i).end(); ++j) {outFileDiff << *j << " ";}
          //       outFileDiff << "\n";            
          //   }
          // }
          // outFileDiff.close();
          // CHECK(correlation == 9);

          // std::string filename_perf = "/hdd1/james/google-research/automl_zero/results_corre3_v3.txt";  
          // std::ofstream outFileperf(filename_perf);
          // outFileperf << ", strategy_ret: ";
          // for (auto i = strategy_ret->begin(); i != strategy_ret->end(); ++i) {
          //   outFileperf << (*i) << ", ";
          // }
          // outFileperf << ". valid_strategy_ret: ";
          // for (auto i = valid_strategy_ret->begin(); i != valid_strategy_ret->end(); ++i) {
          //   outFileperf << (*i) << ", ";
          // }
          // outFileperf.close();
          // CHECK(existing_corre1 == 0.99);

        } else {
          combined_fitness = kMinFitness;
        }

        /// james: cancel write to file only write algo since it's easily to replicate preds later; below is write to file 

        // if (is_search == 1) {
        //   if (combined_fitness > best_select_fitness) { // James: question why sometimes correlation doesn't cheange but here I only allow fitness that is better than best_fitness??
        //     std::ofstream outFile("/hdd1/james/google-research/automl_zero/search_my_file_dde.txt");
        //       for (auto i = all_task_preds.begin(); i != all_task_preds.end(); ++i) {
        //       for (auto j = (*i).begin(); j != (*i).end(); ++j) {outFile << *j << " ";}
        //       outFile << "\n";
        //     }
        //     outFile.close();
        //   }
        // } else if (is_search == 2) {
        //   std::ofstream outFileEval("/hdd1/james/google-research/automl_zero/final_eval_my_file_dde.txt");
        //     for (auto i = all_task_preds.begin(); i != all_task_preds.end(); ++i) {
        //     for (auto j = (*i).begin(); j != (*i).end(); ++j) {outFileEval << *j << " ";}
        //     outFileEval << "\n";
        //   }
        //   outFileEval.close();
        // }
        functional_cache_->InsertOrDie(hash, std::make_pair(combined_fitness, result_vector));
        return std::make_pair(combined_fitness, result_vector);
      }
    }
  }
}

double Evaluator::CorrelationWithExisting(std::vector<double>* valid_strategy_ret, std::vector<double> existing_alpha_ret, IntegerT all_task_preds_size, IntegerT existing_alpha_ret_test_size) {
  IntegerT length = std::min(valid_strategy_ret->size(), existing_alpha_ret.size());

  if (valid_strategy_ret->size() > existing_alpha_ret.size()) {
    vector<double>::const_iterator first_existing_alpha_ret = existing_alpha_ret.begin();
    // cout << "valid_strategy_ret->size()" << valid_strategy_ret->size() << endl;
    // cout << "existing_alpha_ret.size()" << existing_alpha_ret.size() << endl;
    // cout << "existing_alpha_ret_test.size()" << existing_alpha_ret_test.size() << endl;
    vector<double>::const_iterator last_existing_alpha_ret = existing_alpha_ret.begin() + (valid_strategy_ret->size() - (all_task_preds_size - existing_alpha_ret.size() - existing_alpha_ret_test_size));

    vector<double> new_existing_alpha_ret_valid(first_existing_alpha_ret, last_existing_alpha_ret);   

    // for (std::vector<double>::size_type j = 0; j < new_existing_alpha_ret_valid.size(); j++) {
    //   std::cout << "new_existing_alpha_ret_valid[" << j << "]: " << new_existing_alpha_ret_valid[j] << std::endl;
    // }

    double correlation_with_existing_alpha = CorrelationVec(new_existing_alpha_ret_valid, *valid_strategy_ret); 
    return correlation_with_existing_alpha;
   
  } else {
    vector<double>::const_iterator first_valid_strategy_ret = valid_strategy_ret->begin();
    // cout << "existing_alpha_ret.size()" << existing_alpha_ret.size() << endl;
    // cout << "all_task_preds[0].size()" << all_task_preds[0].size() << endl;
    vector<double>::const_iterator last_valid_strategy_ret = valid_strategy_ret->begin() + (existing_alpha_ret.size() - (existing_alpha_ret.size() + existing_alpha_ret_test_size - all_task_preds_size));
    vector<double> new_valid_strategy_ret_valid(first_valid_strategy_ret, last_valid_strategy_ret);
    // for (std::vector<double>::size_type j = 0; j < new_valid_strategy_ret_valid.size(); j++) {
    //   std::cout << "new_valid_strategy_ret_valid[" << j << "]: " << new_valid_strategy_ret_valid[j] << std::endl;
    // }
    double correlation_with_existing_alpha = CorrelationVec(existing_alpha_ret, new_valid_strategy_ret_valid); 
    return correlation_with_existing_alpha;
  }
}

// /// james: check if previous instruction.out_ has the op previous_rank's instruction.in1_
// bool Evaluator::CheckHasIn(const Algorithm* algorithm, IntegerT ins_count, IntegerT in1) {
//   // cout << "check algo" << endl;
//   bool return_type;
//   vector<double> list_int_op = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,21,27,34,44,47,50,51,54,55,56,59,62,65,66,67,71,72,74,75};

//   for (const std::shared_ptr<const Instruction>& myinstruction :
//    algorithm->predict_) {     
//       // cout << "check instruction: " << myinstruction->ToString() << endl;
//     bool found = (std::find(list_int_op.begin(), list_int_op.end(), myinstruction->op_) != list_int_op.end());
//       if (myinstruction->out_ == in1 && found) {
//         // cout << "myinstruction->out_" << myinstruction->out_ << endl;
//         // cout << "in1" << in1 << endl;
//         // cout << "ins_count_rank" << ins_count_rank << endl;
//         // cout << "ins_count" << ins_count << endl;

//         return_type = true;
//         return return_type;
//       } else return_type = false;
//    }

//   for (const std::shared_ptr<const Instruction>& myinstruction :
//    algorithm->learn_) {     
//       // cout << "check instruction: " << myinstruction->ToString() << endl;
//     bool found = (std::find(list_int_op.begin(), list_int_op.end(), myinstruction->op_) != list_int_op.end());
//       if (myinstruction->out_ == in1 && found) {
//         // cout << "myinstruction->out_" << myinstruction->out_ << endl;
//         // cout << "in1" << in1 << endl;
//         // cout << "ins_count_rank" << ins_count_rank << endl;
//         // cout << "ins_count" << ins_count << endl;

//         return_type = true;
//         return return_type;
//       } else return_type = false;
//    }

//   return return_type;
// }

/// james: check if previous instruction.out_ has the op previous_rank's instruction.in1_
bool Evaluator::CheckHasIn(const Algorithm* algorithm, IntegerT ins_type, IntegerT out, vector<IntegerT>* useful_list, std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>* check_cycle_list, IntegerT main_pos, IntegerT if_predict_only) {
  // cout << "check algo" << endl;
  vector<double> list_int_op = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,19,29,44,47,65,66,72,73,75};
  vector<double> list_int_op2 = {1,2,3,4,44,47};
  vector<double> list_int_op_out = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,21,27,34,44,47,50,51,54,55,56,59,62,65,66,67,71,72,74,75};

  vector<double> list_vec_op = {16,20,21,22,23,24,25,26,27,28,32,33,45,48,50,54,71,74};
  vector<double> list_vec_op2 = {18,23,24,25,26,27,28,31,45,48,71,74};
  vector<double> list_vec_op_out = {16,18,19,20,22,23,24,25,26,31,35,36,45,48,52,53,57,60,63,68,69,73};

  vector<double> list_mat_op = {17,30,31,34,35,36,37,38,39,40,41,42,43,46,49,51,52,53,55};
  vector<double> list_mat_op2 = {29,39,40,41,42,43,46,49};
  vector<double> list_mat_op_out = {17,28,29,30,32,33,37,38,39,40,41,42,43,46,49,58,61,64};

  vector<IntegerT> assign_ops = {56,57,58,59,60,61,62,63,64};
  vector<IntegerT> valid_ops = {67,68,69};

  std::vector<std::shared_ptr<const Instruction>> predict_plus_learn;

  // james: insert .learn_ first because instructions in predict_ is nearest to s1
  if (if_predict_only == 0) predict_plus_learn.insert( predict_plus_learn.end(), algorithm->learn_.begin(), algorithm->learn_.end() );
  predict_plus_learn.insert( predict_plus_learn.end(), algorithm->predict_.begin(), algorithm->predict_.end() );

  IntegerT pos = -1;
  IntegerT ins_count_find = 0;
  // cout << "is empty? " << check_cycle_list->empty() << endl;
  if (check_cycle_list->empty()) {
    // james: find the position of last s1
    for (const std::shared_ptr<const Instruction>& myinstruction :
     predict_plus_learn) {    
      bool found3 = (std::find(list_int_op_out.begin(), list_int_op_out.end(), myinstruction->op_) != list_int_op_out.end());
      if (found3 && myinstruction->out_ == 1) {
        pos = ins_count_find;
      }
      ++ins_count_find;
    }     

    // cout << "ins_type: " << ins_type << endl;
    // cout << "out: " << out << endl;
    // cout << "pos: " << pos << endl;
    // if (pos != -1) (*useful_list)[pos] = 1;

    // // std::pair<IntegerT, IntegerT> check_cycle;
    // check_cycle_list->push_back(std::make_pair(out, ins_type));    

  } else {
    for (const std::shared_ptr<const Instruction>& myinstruction :
     predict_plus_learn) {    
      // cout << "is code run here: " << ins_type << "out: " << out << endl; 
      // james: check if the argument is coming from its position in previous instruction in previous call of function's; avoid recheck s1 = s1 + s6
      // bool if_has_iter = false;
      // for (IntegerT i) {
      //   if (ins_count_find == i) {
      //     if_has_iter = true;         
      //   }
      // }

      if (ins_count_find == main_pos) {
        ++ins_count_find;
        continue;
      }

      bool found3 = (std::find(list_int_op_out.begin(), list_int_op_out.end(), myinstruction->op_) != list_int_op_out.end());
      bool found3_vec = (std::find(list_vec_op_out.begin(), list_vec_op_out.end(), myinstruction->op_) != list_vec_op_out.end());
      bool found3_mat = (std::find(list_mat_op_out.begin(), list_mat_op_out.end(), myinstruction->op_) != list_mat_op_out.end());

      switch (ins_type) {
        case 0: {
          if (found3 && myinstruction->out_ == out) {


            if (pos == -1) {
              pos = ins_count_find;
              // cout << "find this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << endl;              
            }

            else if (((0 < (main_pos - ins_count_find)) && ((main_pos - pos) > (main_pos - ins_count_find))) || ((0 > (main_pos - ins_count_find)) && (0 > (main_pos - pos)) && ((main_pos - pos) > (main_pos - ins_count_find)))) {
            pos = ins_count_find;  
            // cout << "find this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << endl;                           
            }

            // IntegerT debug_count = 0;
            // bool if_continue = false;
            // // james: this check is including iteration check because iteration will be included into the check cycle list if pos is found other than -1
            // for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
            //   if ((i->first).first == myinstruction->out_ && (i->first).second == 0 && (i->second).first == ins_count_find && (i->second).second == 0) {
            //     cout << "skipped this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 0 << endl;   
            //     ++debug_count;
            //     CHECK(debug_count < 2);
            //     ++ins_count_find;
            //     if_continue = true;
            //     break;
            //   } else if ((i->first).first == myinstruction->out_ && (i->first).second == 0 && (i->second).first == ins_count_find && (i->second).second == 2) {
            //     cout << "skipped this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 2 << endl;   
            //     ++debug_count;
            //     CHECK(debug_count < 2);
            //     ++ins_count_find;
            //     if_continue = true;
            //     break;
            //   }      
            // }

            // if (if_continue == true) continue;

          }
          break;        
        }
        case 1: {
          if (found3_vec && myinstruction->out_ == out) {
            // IntegerT debug_count = 0;
            // bool if_continue = false;
            // // james: this check is including iteration check because iteration will be included into the check cycle list if pos is found other than -1
            // for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
            //   if ((i->first).first == myinstruction->out_ && (i->first).second == 1 && (i->second).first == ins_count_find && (i->second).second == 0) {
            //     cout << "skipped this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 0 << endl;   
            //     ++ins_count_find;
            //     CHECK(debug_count < 2);
            //     if_continue = true;
            //     break;
            //   } else if ((i->first).first == myinstruction->out_ && (i->first).second == 1 && (i->second).first == ins_count_find && (i->second).second == 2) {
            //     cout << "skipped this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 2 << endl;   
            //     ++ins_count_find;
            //     CHECK(debug_count < 2);
            //     if_continue = true;
            //     break;
            //   }      
            // }

            // if (if_continue == true) continue;

            if (pos == -1) {
                            // cout << "find this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << endl;    
              pos = ins_count_find;            
            }
            
            else if (((0 < (main_pos - ins_count_find)) && ((main_pos - pos) > (main_pos - ins_count_find))) || ((0 > (main_pos - ins_count_find)) && (0 > (main_pos - pos)) && ((main_pos - pos) > (main_pos - ins_count_find)))) {
              pos = ins_count_find;
              // cout << "find this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << endl;  
            }
            
          }
          break;        
        }
        case 2: {
          // IntegerT debug_count = 0;
          // bool if_continue = false;
          // // james: this check is including iteration check because iteration will be included into the check cycle list if pos is found other than -1
          // for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
          //   if ((i->first).first == myinstruction->out_ && (i->first).second == 2 && (i->second).first == ins_count_find && (i->second).second == 0) {
          //     cout << "skipped this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 0 << endl;   
          //     ++ins_count_find;
          //     CHECK(debug_count < 2);
          //     if_continue = true;
          //     break;
          //   } else if ((i->first).first == myinstruction->out_ && (i->first).second == 2 && (i->second).first == ins_count_find && (i->second).second == 2) {
          //     cout << "skipped this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 2 << endl;   
          //     ++ins_count_find;
          //     CHECK(debug_count < 2);
          //     if_continue = true;
          //     break;
          //   }      
          // }

          // if (if_continue == true) continue;

          if (found3_mat && myinstruction->out_ == out) {
            if (pos == -1) {
              // cout << "find this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << endl; 
              pos = ins_count_find;
            }
            
            else if (((0 < (main_pos - ins_count_find)) && ((main_pos - pos) > (main_pos - ins_count_find))) || ((0 > (main_pos - ins_count_find)) && (0 > (main_pos - pos)) && ((main_pos - pos) > (main_pos - ins_count_find)))) {
              pos = ins_count_find;
              // cout << "find this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << endl;  
            }
            
          }   
          break;      
        }
      }

      ++ins_count_find;
    }     

    // cout << "so ins_type: " << ins_type << endl;
    // cout << "and out: " << out << endl;
    // cout << "finally has pos: " << pos << endl;


    // bool if_continue = false;
    // james: this check is including iteration check because iteration will be included into the check cycle list if pos is found other than -1
    for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
      if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second == 0) {
        // cout << "return false this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << (predict_plus_learn[pos])->ToString() << " pos:" << pos << " as: " << 0 << endl;   
        // ++ins_count_find;
        // CHECK(debug_count < 2);
        // if_continue = true;
        return false;
      } else if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second == 2) {
        // cout << "return false this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << (predict_plus_learn[pos])->ToString() << " pos:" << pos << " as: " << 2 << endl;   
        // ++ins_count_find;
        // CHECK(debug_count < 2);
        // if_continue = true;
        return false;
      }      
    }

    // if (if_continue == true) continue;

  }
  
  // james: all below code in this function to get the in1 or in2 of this found instruction. And further check if they are output of other instruction  
  if (pos != -1) {

    const std::shared_ptr<const Instruction>& myinstruction = predict_plus_learn[pos];  

    for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
      if ((i->first).first == myinstruction->out_ && (i->first).second == ins_type && (i->second).first == pos && (i->second).second == 1) {
        // cout << "return true this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 1 << endl;   
        return true;
      }      
    }

    bool found_assign = (std::find(assign_ops.begin(), assign_ops.end(), myinstruction->op_) != assign_ops.end());
    bool found_valid = (std::find(valid_ops.begin(), valid_ops.end(), myinstruction->op_) != valid_ops.end());

    if (found_assign) (*useful_list)[pos] = 1;
    else if (found_valid) {
       // cout << "add here 1" << endl;
      check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 1)));
      (*useful_list)[pos] = 1;
      return true;
    } 
    else {
      // james: assign to unsure 2 first. this is avoid cycle in later check of 1 or 0.
      bool has_cycle_check2 = false;
      for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
        if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second != 2) {
          i->second.second = 2;
          // cout << "init changed this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 2 << endl;   
          has_cycle_check2 = true;
        }     
      }

      if (!has_cycle_check2) {
        check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 2))); 
        // cout << "init saved this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 2 << endl;            
      }

      bool found1 = (std::find(list_int_op.begin(), list_int_op.end(), myinstruction->op_) != list_int_op.end());
      bool found2 = (std::find(list_int_op2.begin(), list_int_op2.end(), myinstruction->op_) != list_int_op2.end());  
      bool found3 = (std::find(list_vec_op.begin(), list_vec_op.end(), myinstruction->op_) != list_vec_op.end());
      bool found4 = (std::find(list_vec_op2.begin(), list_vec_op2.end(), myinstruction->op_) != list_vec_op2.end());  
      bool found5 = (std::find(list_mat_op.begin(), list_mat_op.end(), myinstruction->op_) != list_mat_op.end());
      bool found6 = (std::find(list_mat_op2.begin(), list_mat_op2.end(), myinstruction->op_) != list_mat_op2.end()); 

      // cout << "ins op_ num: " << myinstruction->op_ << endl;
      // cout << "found1" << found1 << "found2" << found2 << "found3" << found3 << "found4" << found4 << "found5" << found5 << "found6" << found6 << endl;

      if (found2 && found1) {
        // cout << "add here 3" << endl;
        // bool skip_1 = false; 
        bool valid_op_1;
        // for (std::pair<IntegerT, IntegerT>q : check_cycle_list) {
        //   if (q.second == 0 && q.first == myinstruction->in1_) skip_1 = true;
        // }

        // if it is the case where s1 = s1 + s5, then s1 should not skip checkhasin
        // if (ins_type == 0 && myinstruction->in1_ == out) {
        // .push_back(pos);
        //   skip_1 = false;
        // }
        if (myinstruction->in1_ == 0) valid_op_1 = true;
        else valid_op_1 = CheckHasIn(algorithm, 0, myinstruction->in1_, useful_list, check_cycle_list, pos, if_predict_only);

        // bool skip_2 = false;
        bool valid_op_2;
        // for (std::pair<IntegerT, IntegerT>q : check_cycle_list) {
        //   if (q.second == 0 && q.first == myinstruction->in2_) skip_2 = true; //// 制造一个新的pair of pair把过去cycle list里的成功失败的ins都标记出来，方便return这个查过的in2的true or false
        // }
        // // if it is the case where s1 = s1 + s5, then s1 should not skip checkhasin
        // if (ins_type == 0 && myinstruction->in2_ == out) {
        // .push_back(pos);
        //   skip_2 = false;
        // }
        if (myinstruction->in2_ == 0) valid_op_2 = true;
        else valid_op_2 = CheckHasIn(algorithm, 0, myinstruction->in2_, useful_list, check_cycle_list, pos, if_predict_only);
        if (!valid_op_2 && !valid_op_1) {

          bool has_cycle = false;
          for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
            if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second != 0) {
              i->second.second = 0;
              // cout << "changed this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 0 << endl;   
              has_cycle = true;
            }     
          }

          if (!has_cycle) {
            check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 0))); 
            // cout << "saved this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 0 << endl;             
          }

          (*useful_list)[pos] = 0;
          return false;
        } else {

          bool has_cycle = false;
          for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
            if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second != 1) {
              i->second.second = 1;
              // cout << "changed this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 1 << endl;   
              has_cycle = true;
            }     
          }

          if (!has_cycle) {
            check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 1))); 
            // cout << "saved this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 1 << endl;             
          }

          (*useful_list)[pos] = 1;
          return true;        
        } 
      } else if (found4 && found1) {
        // cout << "add here 4" << endl;
        // bool skip_1 = false; 
        bool valid_op_1;
        // for (std::pair<IntegerT, IntegerT>q : check_cycle_list) {
        //   if (q.second == 0 && q.first == myinstruction->in1_) skip_1 = true;
        // }
        // // if it is the case where s1 = s1 + s5, then s1 should not skip checkhasin
        // if (ins_type == 0 && myinstruction->in1_ == out) {
        // .push_back(pos);
        //   skip_1 = false;
        // }
        if (myinstruction->in1_ == 0) valid_op_1 = true;
        else valid_op_1 = CheckHasIn(algorithm, 0, myinstruction->in1_, useful_list, check_cycle_list, pos, if_predict_only);

        // bool skip_2 = false;
        bool valid_op_2;
        // for (std::pair<IntegerT, IntegerT>q : check_cycle_list) {
        //   if (q.second == 1 && q.first == myinstruction->in2_) skip_2 = true;
        // }
        // // if it is the case where s1 = s1 + s5, then s1 should not skip checkhasin
        // if (ins_type == 1 && myinstruction->in2_ == out) {
        // .push_back(pos);
        //   skip_2 = false;
        // }
        valid_op_2 = CheckHasIn(algorithm, 1, myinstruction->in2_, useful_list, check_cycle_list, pos, if_predict_only);
        if (!valid_op_2 && !valid_op_1) {

          bool has_cycle = false;
          for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
            if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second != 0) {
              i->second.second = 0;
              // cout << "changed this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 0 << endl;   
              has_cycle = true;
            }     
          }

          if (!has_cycle) {
            check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 0))); 
            // cout << "saved this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 0 << endl;             
          }

          (*useful_list)[pos] = 0;
          return false;
        } else {

          bool has_cycle = false;
          for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
            if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second != 1) {
              i->second.second = 1;
              // cout << "changed this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 1 << endl;   
              has_cycle = true;
            }     
          }

          if (!has_cycle) {
            check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 1))); 
            // cout << "saved this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 1 << endl;             
          }

          (*useful_list)[pos] = 1;
          return true;        
        }  
      } else if (found6 && found1) {
        // cout << "add here 5" << endl;
        // bool skip_1 = false; 
        bool valid_op_1;
        // for (std::pair<IntegerT, IntegerT>q : check_cycle_list) {
        //   if (q.second == 0 && q.first == myinstruction->in1_) skip_1 = true;
        // }
        // // if it is the case where s1 = s1 + s5, then s1 should not skip checkhasin
        // if (ins_type == 0 && myinstruction->in1_ == out) {
        // .push_back(pos);
        //   skip_1 = false;
        // }
        if (myinstruction->in1_ == 0) valid_op_1 = true;
        else valid_op_1 = CheckHasIn(algorithm, 0, myinstruction->in1_, useful_list, check_cycle_list, pos, if_predict_only);

        // bool skip_2 = false;
        bool valid_op_2;
        // for (std::pair<IntegerT, IntegerT>q : check_cycle_list) {
        //   if (q.second == 2 && q.first == myinstruction->in2_) skip_2 = true;
        // }
        // // if it is the case where s1 = s1 + s5, then s1 should not skip checkhasin
        // if (ins_type == 2 && myinstruction->in2_ == out) {
        // .push_back(pos);
        //   skip_2 = false;
        // }
        if (myinstruction->in2_ == 0) valid_op_2 = true;
        else valid_op_2 = CheckHasIn(algorithm, 2, myinstruction->in2_, useful_list, check_cycle_list, pos, if_predict_only);

        if (!valid_op_2 && !valid_op_1) {

          bool has_cycle = false;
          for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
            if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second != 0) {
              i->second.second = 0;
              // cout << "changed this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 0 << endl;   
              has_cycle = true;
            }     
          }

          if (!has_cycle) {
            check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 0))); 
            // cout << "saved this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 0 << endl;             
          }

          (*useful_list)[pos] = 0;
          return false;
        } else {

          bool has_cycle = false;
          for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
            if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second != 1) {
              i->second.second = 1;
              // cout << "changed this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 1 << endl;   
              has_cycle = true;
            }     
          }

          if (!has_cycle) {
            check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 1))); 
            // cout << "saved this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 1 << endl;             
          }

          (*useful_list)[pos] = 1;
          return true;        
        }  
      } else if (found3 && found4) {
        // cout << "add here 6" << endl;
        // bool skip_1 = false; 
        bool valid_op_1;
        // for (std::pair<IntegerT, IntegerT>q : check_cycle_list) {
        //   if (q.second == 1 && q.first == myinstruction->in1_) skip_1 = true;
        // }
        // // if it is the case where s1 = s1 + s5, then s1 should not skip checkhasin
        // if (ins_type == 1 && myinstruction->in1_ == out) {
        // .push_back(pos);
        //   skip_1 = false;
        // }

        valid_op_1 = CheckHasIn(algorithm, 1, myinstruction->in1_, useful_list, check_cycle_list, pos, if_predict_only);

        // bool skip_2 = false;
        bool valid_op_2;
        // for (std::pair<IntegerT, IntegerT>q : check_cycle_list) {
        //   if (q.second == 1 && q.first == myinstruction->in2_) skip_2 = true;
        // }
        // // if it is the case where s1 = s1 + s5, then s1 should not skip checkhasin
        // if (ins_type == 1 && myinstruction->in2_ == out) {
        // .push_back(pos);
        //   skip_2 = false;
        // }
        valid_op_2 = CheckHasIn(algorithm, 1, myinstruction->in2_, useful_list, check_cycle_list, pos, if_predict_only);
        if (!valid_op_2 && !valid_op_1) {

          bool has_cycle = false;
          for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
            if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second != 0) {
              i->second.second = 0;
              // cout << "changed this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 0 << endl;   
              has_cycle = true;
            }     
          }

          if (!has_cycle) {
            check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 0))); 
            // cout << "saved this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 0 << endl;             
          }

          (*useful_list)[pos] = 0;
          return false;
        } else {

          bool has_cycle = false;
          for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
            if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second != 1) {
              i->second.second = 1;
              // cout << "changed this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 1 << endl;   
              has_cycle = true;
            }     
          }

          if (!has_cycle) {
            check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 1))); 
            // cout << "saved this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 1 << endl;             
          }

          (*useful_list)[pos] = 1;
          return true;        
        }  
      } else if (found5 && found6) {
        // cout << "add here 7" << endl;
        // bool skip_1 = false; 
        bool valid_op_1;
        // for (std::pair<IntegerT, IntegerT>q : check_cycle_list) {
        //   if (q.second == 2 && q.first == myinstruction->in1_) skip_1 = true;
        // }
        // // if it is the case where s1 = s1 + s5, then s1 should not skip checkhasin
        // if (ins_type == 2 && myinstruction->in1_ == out) {
        // .push_back(pos);
        //   skip_1 = false;
        // }
        if (myinstruction->in1_ == 0) valid_op_1 = true;
        else valid_op_1 = CheckHasIn(algorithm, 2, myinstruction->in1_, useful_list, check_cycle_list, pos, if_predict_only);

        // bool skip_2 = false;
        bool valid_op_2;
        // for (std::pair<IntegerT, IntegerT>q : check_cycle_list) {
        //   if (q.second == 2 && q.first == myinstruction->in2_) skip_2 = true;
        // }
        // // if it is the case where s1 = s1 + s5, then s1 should not skip checkhasin
        // if (ins_type == 2 && myinstruction->in2_ == out) {
        // .push_back(pos);
        //   skip_2 = false;
        // }
        if (myinstruction->in2_ == 0) valid_op_2 = true;
        else valid_op_2 = CheckHasIn(algorithm, 2, myinstruction->in2_, useful_list, check_cycle_list, pos, if_predict_only);

        if (!valid_op_2 && !valid_op_1) {

          bool has_cycle = false;
          for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
            if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second != 0) {
              i->second.second = 0;
              // cout << "changed this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 0 << endl;   
              has_cycle = true;
            }     
          }

          if (!has_cycle) {
            check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 0))); 
            // cout << "saved this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 0 << endl;             
          }

          (*useful_list)[pos] = 0;
          return false;
        } else {

          bool has_cycle = false;
          for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
            if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second != 1) {
              i->second.second = 1;
              // cout << "changed this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 1 << endl;   
              has_cycle = true;
            }     
          }

          if (!has_cycle) {
            check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 1))); 
            // cout << "saved this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 1 << endl;             
          }

          (*useful_list)[pos] = 1;
          return true;        
        }  
      } else if (found5 && found4) {
        // cout << "add here 7" << endl;
        // bool skip_1 = false; 
        bool valid_op_1;
        // for (std::pair<IntegerT, IntegerT>q : check_cycle_list) {
        //   if (q.second == 2 && q.first == myinstruction->in1_) skip_1 = true;
        // }
        // // if it is the case where s1 = s1 + s5, then s1 should not skip checkhasin
        // if (ins_type == 2 && myinstruction->in1_ == out) {
        // .push_back(pos);
        //   skip_1 = false;
        // }
        if (myinstruction->in1_ == 0) valid_op_1 = true;
        else valid_op_1 = CheckHasIn(algorithm, 2, myinstruction->in1_, useful_list, check_cycle_list, pos, if_predict_only);

        // bool skip_2 = false;
        bool valid_op_2;
        // for (std::pair<IntegerT, IntegerT>q : check_cycle_list) {
        //   if (q.second == 1 && q.first == myinstruction->in2_) skip_2 = true;
        // }
        // // if it is the case where s1 = s1 + s5, then s1 should not skip checkhasin
        // if (ins_type == 1 && myinstruction->in2_ == out) {
        // .push_back(pos);
        //   skip_2 = false;
        // }
        valid_op_2 = CheckHasIn(algorithm, 1, myinstruction->in2_, useful_list, check_cycle_list, pos, if_predict_only);
        if (!valid_op_2 && !valid_op_1) {

          bool has_cycle = false;
          for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
            if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second != 0) {
              i->second.second = 0;
              // cout << "changed this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 0 << endl;   
              has_cycle = true;
            }     
          }

          if (!has_cycle) {
            check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 0))); 
            // cout << "saved this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 0 << endl;             
          }

          (*useful_list)[pos] = 0;
          return false;
        } else {

          bool has_cycle = false;
          for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
            if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second != 1) {
              i->second.second = 1;
              // cout << "changed this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 1 << endl;   
              has_cycle = true;
            }     
          }

          if (!has_cycle) {
            check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 1))); 
            // cout << "saved this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 1 << endl;             
          }

          (*useful_list)[pos] = 1;
          return true;        
        } 
      } else if (found1 && !found2 && !found3 && !found4 && !found5 && !found6) {
        // cout << "add here 8" << endl;
        // bool skip_1 = false; 
        bool valid_op_1;
        // for (std::pair<IntegerT, IntegerT>q : check_cycle_list) {
        //   if (q.second == 0 && q.first == myinstruction->in1_) skip_1 = true;
        // }
        // // if it is the case where s1 = s1 + s5, then s1 should not skip checkhasin
        // if (ins_type == 0 && myinstruction->in1_ == out) {
        // .push_back(pos);
        //   skip_1 = false;
        // }
        if (myinstruction->in1_ == 0) valid_op_1 = true;
        else valid_op_1 = CheckHasIn(algorithm, 0, myinstruction->in1_, useful_list, check_cycle_list, pos, if_predict_only);
        if (!valid_op_1) {

          bool has_cycle = false;
          for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
            if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second != 0) {
              i->second.second = 0;
              // cout << "changed this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 0 << endl;   
              has_cycle = true;
            }     
          }

          if (!has_cycle) {
            check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 0))); 
            // cout << "saved this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 0 << endl;             
          }

          (*useful_list)[pos] = 0;
          return false;
        } else {

          bool has_cycle = false;
          for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
            if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second != 1) {
              i->second.second = 1;
              // cout << "changed this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 1 << endl;   
              has_cycle = true;
            }     
          }

          if (!has_cycle) {
            check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 1))); 
            // cout << "saved this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 1 << endl;             
          }

          (*useful_list)[pos] = 1;
          return true;        
        } 
      } else if (!found1 && !found2 && found3 && !found4 && !found5 && !found6) {
        // bool skip = false;
        // for (std::pair<IntegerT, IntegerT>q : check_cycle_list) {
        //   if (q.second == 1 && q.first == myinstruction->in1_) skip = true;
        // }
        // // if it is the case where s1 = s1 + s5, then s1 should not skip checkhasin
        // if (ins_type == 1 && myinstruction->in1_ == out) {
        // .push_back(pos);
        //   skip = false;
        // }
        // cout << "found3 here" << endl;  
        bool valid_op_1;
        valid_op_1 = CheckHasIn(algorithm, 1, myinstruction->in1_, useful_list, check_cycle_list, pos, if_predict_only);
        if (!valid_op_1) {
          bool has_cycle = false;
          for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
            if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second != 0) {
              i->second.second = 0;
              // cout << "changed this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 0 << endl;   
              has_cycle = true;
            }     
          }

          if (!has_cycle) {
            check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 0))); 
            // cout << "saved this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 0 << endl;             
          }

          (*useful_list)[pos] = 0;
          return false;
        } else {

          bool has_cycle = false;
          for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
            if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second != 1) {
              i->second.second = 1;
              // cout << "changed this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 1 << endl;   
              has_cycle = true;
            }     
          }

          if (!has_cycle) {
            check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 1))); 
            // cout << "saved this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 1 << endl;             
          }

          (*useful_list)[pos] = 1;
          return true;        
        }       
      } else if (!found1 && !found2 && !found3 && !found4 && found5 && !found6) {
        bool valid_op_1;     
        // bool skip = false;
        // for (std::pair<IntegerT, IntegerT>q : check_cycle_list) {
        //   if (q.second == 2 && q.first == myinstruction->in1_) skip = true; 
        // }
        // // if it is the case where s1 = s1 + s5, then s1 should not skip checkhasin
        // if (ins_type == 2 && myinstruction->in1_ == out) {
        // .push_back(pos);
        //   skip = false;
        // }
        if (myinstruction->in1_ == 0) valid_op_1 = true; 
        else valid_op_1 = CheckHasIn(algorithm, 2, myinstruction->in1_, useful_list, check_cycle_list, pos, if_predict_only);
        if (!valid_op_1) {
          bool has_cycle = false;
          for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
            if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second != 0) {
              i->second.second = 0;
              // cout << "changed this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 0 << endl;   
              has_cycle = true;
            }     
          }

          if (!has_cycle) {
            check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 0))); 
            // cout << "saved this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 0 << endl;             
          }

          (*useful_list)[pos] = 0;
          return false;
        } else {

          bool has_cycle = false;
          for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
            if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second != 1) {
              i->second.second = 1;
              // cout << "changed this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 1 << endl;   
              has_cycle = true;
            }     
          }

          if (!has_cycle) {
            check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 1))); 
            // cout << "saved this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 1 << endl;             
          }

          (*useful_list)[pos] = 1;
          return true;        
        }
      }

      // for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
      //   if ((i->first).first == myinstruction->out_ && (i->first).second == ins_type && (i->second).first == -1 && (i->second).second == 0) {
      //     if (CheckHasIn(algorithm, (i->first).second, (i->first).first, useful_list, check_cycle_list, pos, if_predict_only))
      //     cout << "return true this type: " << ins_type << "(" << out << ") in this " << "instruction here: " << myinstruction->ToString() << " pos:" << pos << " as: " << 1 << endl;   
      //     return true;
      //   }      
      // }    

    }   
  } else {
    // cout << "add here 9" << endl;
      // check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(main_pos, 0))); 
      // cout << "saved 0 this type: " << ins_type << "(" << out << " maiin_pos:" << main_pos << " as: " << 0 << endl; 
      // (*useful_list)[pos] = 0;
      return false;
  } 
}

bool Evaluator::CheckCompleteness(const std::vector<std::vector<double> > data) {
  for (auto i = data.begin(); i != data.end(); ++i) {
    if ((*i).size() == 0) {
      return false;
    }
  }
  return true;
}
std::vector<std::vector<double> > Evaluator::Transpose(const std::vector<std::vector<double> > data) {
    // this assumes that all inner vectors have the same size and
    // allocates space for the complete result in advance
    // IntegerT num_dim = 0;
    // for (auto j = (*all_task_preds.begin()).begin(); j != (*all_task_preds.begin()).end(); ++j) {++num_dim;}
    // vector<vector<double>> result;
    // result.clear();
    // result.resize(data[0].size());
    // for (vector<double>& value : result) {
    //   value.resize(data.size(), 1);
    // }
    std::vector<std::vector<double> > result(data[0].size(),
                                          std::vector<double>(data.size()));
    // cout << " data[0].size() " << data[0].size() << endl;
    // cout << " data[21].size() " << data[21].size() << endl;
    // cout << " data[22].size() " << data[22].size() << endl;
    // cout << " data[23].size() " << data[23].size() << endl;
    // cout << " data[24].size() " << data[24].size() << endl;
    // cout << " data.size() " << data.size() << endl;
    for (std::vector<double>::size_type i = 0; i < data[0].size(); i++) 
        for (std::vector<double>::size_type j = 0; j < data.size(); j++) {
            // cout << " i " << i << endl;
            // cout << " j " << j << endl;
          // if (i == 0 && j == 22) {
          //   cout << " data[21][0] " << data[21][i] << endl;
          //   cout << " data[22][0] " << data[j][i] << endl;
          // }
            result[i][j] = data[j][i];
            // cout << " result[i][j] " << result[i][j] << endl;
            // cout << " data[j][i] " << data[j][i] << endl;
        }
    return result;
}

std::vector<std::vector<double> > Evaluator::GetValidTest(const std::vector<std::vector<double> > data, bool is_validate) {
  int index =  static_cast<int>(data[0].size()/2);
  // cout << "index: " << index << endl;

  std::vector<std::vector<double> > result(data.size(),
                                        std::vector<double>(index));
  // cout << "data.size()" << data.size() << endl;
  // cout << " code run here 4? " << endl;
  std::vector<std::vector<double> > result_not_validate(data.size(),
                                        std::vector<double>(data[0].size() - index));
  // cout << " code run here 5? " << endl;
  if (is_validate) {
    // cout << " code run here 6? " << endl;
    for (std::vector<double>::size_type i = 0; i < data.size(); i++)
      {
        // cout << " code run here 7? " << endl;
        for (std::vector<double>::size_type j = 0; j < index; j++) {
          // if (data[i].empty()) cout << "empty" << endl;
          // cout << "i: " << i << endl;
          // cout << "j: " << j << endl;
          result[i][j] = data[i][j];
        }
      }
      return result;      
  } else {
    // cout << " code run here 8? " << endl;
    for (std::vector<double>::size_type i = 0; i < data.size(); i++)
      {
        // cout << " code run here 41? " << endl;
        for (std::vector<double>::size_type j = index; j < data[0].size(); j++) {
          // cout << " code run here 42? " << endl;
          result_not_validate[i][j-index] = data[i][j];
        }
      }
      return result_not_validate;      
  }
}

struct myclasslarge {
  bool operator() (std::pair<double, int> &i, std::pair<double, int> &j) { return (i.first > j.first);} /// james: if all preds are equal then don't change sequence
} myobjectlarge;
// maxindices.cc
// compile with:
// g++ -std=c++11 maxindices.cc -o maxindices
std::vector<IntegerT> Evaluator::TopkLarge(const std::vector<double> test, const int k) {
  std::vector<std::pair<double, int>> q;
  std::vector<IntegerT> result;
  // result.resize(k);
  for (int i = 0; i < test.size(); ++i) {
    q.push_back(std::pair<double, int>(test[i], i));
  }
  // std::cout << "top[" << i << "] = " << ki << std::endl;
  std::stable_sort(q.begin(), q.end(), automl_zero::operatorf);
  // int k = 3; // number of indices we need
  for (int i = 0; i < k; ++i) {
    int ki = q[i].second;
    // std::cout << "top[" << i << "] = " << ki << std::endl;
    // std::cout << "top[" << i << "] = " << q[i].first << std::endl;
    result.push_back(ki);
  }
  return result;
}

struct myclass {
  bool operator() (std::pair<double, int> &i, std::pair<double, int> &j) { return (i.first < j.first);}
} myobject;

std::vector<IntegerT> Evaluator::TopkSmall(const std::vector<double> test, const int k) {
  std::vector<std::pair<double, int>> q;
  std::vector<IntegerT> result;
  // result.resize(k);
  for (int i = 0; i < test.size(); ++i) {
    q.push_back(std::pair<double, int>(test[i], i));
  }
  // std::cout << "top[" << i << "] = " << ki << std::endl;
  std::stable_sort(q.begin(), q.end());
  // for (int i = 0; i < test.size(); ++i) {
  //   // std::cout << "after sort bottom second[" << i << "] = " << q[i].second << std::endl;
  //   // std::cout << "after sort bottom first[" << i << "] = " << q[i].first << std::endl;
  // }
  // int k = 3; // number of indices we need
  for (int i = 0; i < k; ++i) {
    int ki = q[i].second;
    // std::cout << "bottom[" << i << "] = " << ki << std::endl;
    // std::cout << "bottom[" << i << "] = " << q[i].first << std::endl;
    result.push_back(ki);
  }
  // std::cout << "test.size()" << test.size() << std::endl;
  // CHECK(k == 999);
  return result;
}

double Evaluator::ComputeAllMeasure(const std::vector<std::vector<double> > all_task_preds, const std::vector<std::vector<double> > price_diff, double* sharpe_ratio, double* average_holding_days, double* strat_ret_vol, double* annual_mean_strat_ret, std::vector<double>* strategy_ret, bool is_validate) {
    CHECK(all_task_preds.size() == price_diff.size());
    for (std::vector<double>::size_type j = 0; j < all_task_preds.size(); j++) {
      CHECK(all_task_preds[j].size() == price_diff[j].size());
    }
    IntegerT num_holding = 50;
    std::vector<double> preds_column;
    std::vector<double> price_diff_column;
    std::vector<IntegerT> top100_preds_index;
    std::vector<IntegerT> bottom100_preds_index;
    top100_preds_index.resize(num_holding);
    bottom100_preds_index.resize(num_holding);
    std::vector<IntegerT> holding_stocks; // stocks currently holding, stocks are represented as stock indices 
    std::vector<IntegerT> holding_stocks_short; // stocks currently holding short, stocks are represented as stock indices 
    std::vector<double> holding_stocks_shares;
    std::vector<double> holding_stocks_shares_short;
    std::vector<IntegerT> holding_days;
    std::vector<IntegerT> holding_days_short;
    std::vector<double> strategy_returns;
    double NAV = 2000000.0;
    double old_NAV = 2000000.0;
    double cash = 2000000.0;
    double init_cap = 1000000.0; /// cash flow in and out when buying and selling stocks
    double init_cap_short = 1000000.0;
    double old_balance = 1000000.0;    
    double old_balance_short = 1000000.0; 
    double new_balance = 1000000.0;    
    double new_balance_short = 1000000.0;   
    double dropdown_strat_ret_accu = 1.0;
    double lowest_dropdown_strat_ret_accu = 1.0;
    IntegerT holding_days_total = 0;
    IntegerT holding_counts = 0;    

    // loop over column and then over rows 
    for (std::vector<double>::size_type i = 0; i < all_task_preds[0].size(); i++) 
      {
        preds_column.clear();
        price_diff_column.clear();

        CHECK(preds_column.size() == 0);
        CHECK(price_diff_column.size() == 0);
        for (std::vector<double>::size_type j = 0; j < all_task_preds.size(); j++) {
          if ((std::abs(all_task_preds[j][i]) != 1234 && std::abs(price_diff[j][i]) == 1234) || (std::abs(all_task_preds[j][i]) == 1234 && std::abs(price_diff[j][i]) != 1234)) {
            cout << " why not together? " << preds_column[j] << endl;
            cout << " why not together? j+1 " << preds_column[j+1] << endl;
            cout << " j: " << j << endl;
            cout << " i: " << i << endl;
            for (std::vector<double>::size_type k = 0; k < all_task_preds.size(); k++) {
              cout << " price_diff[k][i] " << price_diff[k][i] << endl;
              cout << " preds_column[k] " << preds_column[k] << endl;
              if (k == 30) break;
            }
            cout << " all_task_preds[j-1][i] " << all_task_preds[j-1][i] << endl;
            cout << " price_diff[j-1][i] " << price_diff[j-1][i] << endl;
            cout << " all_task_preds[j][i] " << all_task_preds[j][i] << endl;
            cout << " price_diff[j][i] " << price_diff[j][i] << endl;
            cout << " all_task_preds[j+1][i] " << all_task_preds[j+1][i] << endl;
            cout << " price_diff[j+1][i] " << price_diff[j+1][i] << endl;
            cout << " all_task_preds[j+2][i] " << all_task_preds[j+2][i] << endl;
            cout << " price_diff[j+2][i] " << price_diff[j+2][i] << endl;
            cout << " all_task_preds[j][i-1] " << all_task_preds[j][i-1] << endl;
            cout << " price_diff[j][i-1] " << price_diff[j][i-1] << endl;
            cout << " all_task_preds[j][i+1] " << all_task_preds[j][i+1] << endl;
            cout << " price_diff[j][i+1] " << price_diff[j][i+1] << endl;
            cout << " all_task_preds[j][i+2] " << all_task_preds[j][i+2] << endl;
            cout << " price_diff[j][i+2] " << price_diff[j][i+2] << endl;
            // // write to file
            // std::ofstream outFile("/hdd1/james/google-research/automl_zero/train_my_file_1.txt");
            //   for (auto i = all_task_preds.begin(); i != all_task_preds.end(); ++i) {
            //   for (auto j = (*i).begin(); j != (*i).end(); ++j) {outFile << *j << " ";}
            //   outFile << "\n";
            // }
            // std::ofstream outFileDiff("/hdd1/james/google-research/automl_zero/train_my_file_diff.txt");
            //   for (auto i = price_diff.begin(); i != price_diff.end(); ++i) {
            //   for (auto j = (*i).begin(); j != (*i).end(); ++j) {outFileDiff << *j << " ";}
            //   outFileDiff << "\n";
            // }
            CHECK(j == 1234);
          }
          if (std::abs(all_task_preds[j][i]) != 1234 && std::abs(price_diff[j][i]) != 1234) {
            preds_column.push_back(all_task_preds[j][i]);
            price_diff_column.push_back(price_diff[j][i]);

            // std::cout << "all_prices[j][i] " << all_prices[j][i] << std::endl;
            // cout << " price_diff[j][i]: " << price_diff[j][i] << endl;
            // CHECK(i != 2);
            if (std::abs(all_task_preds[j][i] * price_diff[j][i]) > 1) {
              cout << " why larger than 1? " << all_task_preds[j][i] << endl;
            }
          } 
          // cout << " result[i][j] " << result[i][j] << endl;
          // cout << " data[j][i] " << data[j][i] << endl;
        }
        if (!holding_stocks.empty()) CHECK(holding_stocks.size() == num_holding);
        if (!holding_stocks_shares.empty()) CHECK(holding_stocks_shares.size() == num_holding);
        if (!holding_days.empty()) CHECK(holding_days.size() == num_holding);

        std::vector<IntegerT> stocks_to_buy;
        std::vector<IntegerT> stocks_to_sell;
        std::vector<IntegerT> stocks_to_short;
        std::vector<IntegerT> bottom_stocks_to_exclude;
        stocks_to_buy.clear();
        stocks_to_sell.clear();
        stocks_to_short.clear();
        bottom_stocks_to_exclude.clear();

        stocks_to_buy = TopkLarge(preds_column, num_holding);
        stocks_to_short = TopkSmall(preds_column, num_holding);
        // QUESTION: below code is to find sum of alphas. should I use the weight of alpha to decide how much to buy or just buy each stock evenly?
        // double sum_top_preds = 0;
        // for (std::vector<double>::size_type j = 0; j < top100_preds_index.size(); j++) {
        //   sum_top_preds += prices_column[top100_preds_index[j]];
        // }

        // cout << "stocks_to_buy[" << endl;
        // for (std::vector<double>::size_type j = 0; j < stocks_to_buy.size(); j++) {
        //   cout << "stocks_to_buy[" << j << "]: " << stocks_to_buy[j] << endl;
        // }
        // for (std::vector<double>::size_type j = 0; j < stocks_to_short.size(); j++) {
        //   cout << "stocks_to_short[" << j << "]: " << stocks_to_short[j] << endl;
        // }

        // if in top100_preds_index list don't have the one in loop of holding_stocks then sell the one
        // std::cout << "holding_stocks.size() 4 " << holding_stocks.size() << std::endl;

        if (!holding_stocks.empty()) {
          
          for (std::vector<double>::size_type j = 0; j < holding_stocks.size(); j++) {
            // cout << "holding_stocks[" << j << "]: " << holding_stocks[j] << endl;
            if (std::find(stocks_to_buy.begin(), stocks_to_buy.end(), holding_stocks[j]) == stocks_to_buy.end()) 
              stocks_to_sell.push_back(holding_stocks[j]);
              // cout << "stocks_to_sell[" << j << "]: " << holding_stocks[j] << endl;
          }          
        }

        /// holding_stocks_short 
        if (!holding_stocks_short.empty()) {
          for (std::vector<double>::size_type j = 0; j < holding_stocks_short.size(); j++) {
            // cout << "holding_stocks_short[" << j << "]: " << holding_stocks_short[j] << endl;
            if (std::find(stocks_to_short.begin(), stocks_to_short.end(), holding_stocks_short[j]) == stocks_to_short.end()) 
              bottom_stocks_to_exclude.push_back(holding_stocks_short[j]);
              // cout << "bottom_stocks_to_exclude[" << j << "]: " << holding_stocks_short[j] << endl;
          }          
        }

        // if in holding_stocks list don't have the one in loop of top100_preds_index then buy the one
        // for (std::vector<double>::size_type j = 0; j < top100_preds_index.size(); j++) {
        //   if (std::find(holding_stocks.begin(), holding_stocks.end(), top100_preds_index[j]) == holding_stocks.end()) 
        //     stocks_to_buy.push_back(top100_preds_index[j]);
        //     if (top100_preds_index[j] > 10000) cout << " top100_preds_index[j] " << top100_preds_index[j] << endl;
        // }

        // // if in holding_stocks list don't have the one in loop of top100_preds_index then buy the one
        // for (std::vector<double>::size_type j = 0; j < bottom100_preds_index.size(); j++) {
        //   if (std::find(holding_stocks_short.begin(), holding_stocks_short.end(), bottom100_preds_index[j]) == holding_stocks_short.end()) 
        //      stocks_to_short.push_back(bottom100_preds_index[j]);
        //     if (top100_preds_index[j] > 10000) cout << " bottom100_preds_index[j] " << top100_preds_index[j] << endl;
        // }

        // cout << " stock to buy " << stocks_to_buy[98] << endl;
        // if stocks not in sell list then add holding_days for the holding stock
        // if (!holding_stocks.empty()) {
        //   for (std::vector<double>::size_type j = 0; j < holding_stocks.size(); j++) {
        //     if (std::find(stocks_to_sell.begin(), stocks_to_sell.end(), holding_stocks[j]) == stocks_to_sell.end()) {
        //       holding_days[j] += 1;
        //     } 
        //   }
        // }        

        // if stocks not in exclude list then add holding_days for the short stock
        // if (!holding_stocks_short.empty()) {
        //   for (std::vector<double>::size_type j = 0; j < holding_stocks_short.size(); j++) {
        //     if (std::find(bottom_stocks_to_exclude.begin(), bottom_stocks_to_exclude.end(), holding_stocks_short[j]) == bottom_stocks_to_exclude.end()) {
        //       holding_days_short[j] += 1;
        //     } 
        //   }
        // } 

        // sell the stocks and add money
        if (!holding_stocks.empty()) {        
          for (std::vector<double>::size_type j = 0; j < stocks_to_sell.size(); j++) {
            std::vector<IntegerT>::iterator it = std::find(holding_stocks.begin(), holding_stocks.end(), stocks_to_sell[j]);
            int index = std::distance(holding_stocks.begin(), it);
            // std::vector<double>::iterator it_shares = std::find(holding_stocks_shares.begin(), holding_stocks_shares.end(), holding_stocks_shares[index]);
            // cout << " code run here 4? " << endl;
            // std::vector<IntegerT>::iterator it_days = std::find(holding_days.begin(), holding_days.end(), holding_days[index]);          
            if (it != holding_stocks.end()) {
              // cout << "erase holding stock index[" << index << "]: " << holding_stocks[index] << endl;
              holding_stocks.erase(it);
              // cout << " code run here 5? " << endl;
              /// James: '''change from holding shares to just holding money for simplicity. The difference with holding shares is just minimum hands of shares to hold (e.g. 100 shares or integer shares)''';
              // init_cap += holding_stocks_shares[index] * prices_column[stocks_to_sell[j]];
              // cout << "erase holding stock shares[" << index << "]: " << holding_stocks_shares[index] << endl;
              // init_cap += holding_stocks_shares[index];
              // cout << " code run here 7? " << endl;
              holding_stocks_shares.erase(holding_stocks_shares.begin()+index);
              // cout << " code run here 8? " << endl;
              holding_counts += 1;
              // cout << "erase holding stock days: " << holding_days[index] << endl;
              holding_days_total += holding_days[index];
              // cout << " code run here 6? " << endl;
              holding_days.erase(holding_days.begin()+index);

            } else {
              std::cout << "why cannot find stocks_to_sell in holding_stocks??" << std::endl;
            }
          }
        }
        // std::cout << "holding_stocks.size() 1 " << holding_stocks.size() << std::endl;

        // exclude the stocks not in short list and add quota available to short
        if (!holding_stocks_short.empty()) {        
          for (std::vector<double>::size_type j = 0; j < bottom_stocks_to_exclude.size(); j++) {
            // cout << "bottom_stocks_to_exclude[" << j << "]: " << bottom_stocks_to_exclude[j] << endl;
            std::vector<IntegerT>::iterator it = std::find(holding_stocks_short.begin(), holding_stocks_short.end(), bottom_stocks_to_exclude[j]);
            int index = std::distance(holding_stocks_short.begin(), it);
            // std::vector<double>::iterator it_shares = std::find(holding_stocks_shares.begin(), holding_stocks_shares.end(), holding_stocks_shares[index]);
            // cout << " code run here 4? " << endl;
            // std::vector<IntegerT>::iterator it_days = std::find(holding_days.begin(), holding_days.end(), holding_days[index]);          
            if (it != holding_stocks_short.end()) {
              // cout << "erase holding stock short index[" << index << "]: " << holding_stocks_short[index] << endl;
              holding_stocks_short.erase(it);
              // cout << " code run here 5? " << endl;
              /// James: '''change from holding shares to just holding money for simplicity. The difference with holding shares is just minimum hands of shares to hold (e.g. 100 shares or integer shares)''';
              // init_cap += holding_stocks_shares[index] * prices_column[stocks_to_sell[j]];
              // init_cap_short += holding_stocks_shares_short[index];
              // cout << " code run here 7? " << endl;
              holding_stocks_shares_short.erase(holding_stocks_shares_short.begin()+index);
              // cout << " code run here 8? " << endl;
              holding_counts += 1;
              holding_days_total += holding_days_short[index];
              // cout << " code run here 6? " << endl;
              holding_days_short.erase(holding_days_short.begin()+index);
            } else {
              std::cout << "why cannot find stocks_to_sell in holding_stocks??" << std::endl;
            }
          }
        }
        // std::cout << "holding_stocks_short.size() " << holding_stocks_short.size() << std::endl;
        // James: if no stocks to buy or to sell then the algo is not predicting anything
        // if (holding_stocks.size() == 100) {
        //   std::cout << "stocks_to_buy.size " << stocks_to_buy.size() << std::endl;
        //   std::cout << "stocks_to_sell.size " << stocks_to_sell.size() << std::endl;
        //   for (std::vector<double>::size_type j = 0; j < stocks_to_buy.size(); j++) {
        //     std::cout << "stocks_to_buy[j] " << stocks_to_buy[j] << std::endl;
        //   }
        //   for (std::vector<double>::size_type j = 0; j < stocks_to_sell.size(); j++) {
        //     std::cout << "stocks_to_sell[j] " << stocks_to_sell[j] << std::endl;
        //   }
        //   for (std::vector<double>::size_type j = 0; j < holding_stocks.size(); j++) {
        //     std::cout << "holding_stocks[j] " << holding_stocks[j] << std::endl;
        //   }
        //   for (std::vector<double>::size_type j = 0; j < top100_preds_index.size(); j++) {
        //     std::cout << "top100_preds_index[j] " << top100_preds_index[j] << std::endl;
        //   }
        //   for (std::vector<double>::size_type j = 0; j < top100_preds_index.size(); j++) {
        //     std::cout << "top100_preds_index[j] " << preds_column[top100_preds_index[j]] << std::endl;
        //   }
        //   std::cout << "holding_stocks.size() 6 " << holding_stocks.size() << std::endl;
        //   CHECK(holding_stocks.size() == 100);
        //   // if (!holding_stocks.empty()) {
            // cout << " code run here 61? " << endl;
        //   //   for (std::vector<double>::size_type j = 0; j < holding_stocks.size(); j++) {
              // cout << " code run here 62? " << endl;
        //   //     if (std::find(top100_preds_index.begin(), top100_preds_index.end(), holding_stocks[j]) == top100_preds_index.end()) {
                // cout << " code run here 63? " << endl;
        //   //       stocks_to_sell.push_back(holding_stocks[j]);
        //   //     }
        //   //   }          
        //   // }

        //   // if in holding_stocks list don't have the one in loop of top100_preds_index then buy the one
        //   // for (std::vector<double>::size_type j = 0; j < top100_preds_index.size(); j++) {
            // cout << " code run here 64? " << endl;
        //   //   if (std::find(holding_stocks.begin(), holding_stocks.end(), top100_preds_index[j]) == holding_stocks.end()) {
              // cout << " code run here 65? " << endl;
        //   //     stocks_to_buy.push_back(top100_preds_index[j]);
        //   //     if (top100_preds_index[j] > 10000) cout << " top100_preds_index[j] " << top100_preds_index[j] << endl;
        //   //   }
        //   // }
        //   // cout << " stock to buy " << stocks_to_buy[98] << endl;
        //   // if stocks not in sell list then add holding_days for the holding stock
        //   // if (!holding_stocks.empty()) {
            // cout << " code run here 66? " << endl;
        //   //   for (std::vector<double>::size_type j = 0; j < holding_stocks.size(); j++) {
        //   //     if (std::find(stocks_to_sell.begin(), stocks_to_sell.end(), holding_stocks[j]) == stocks_to_sell.end()) {
        //   //       holding_days[j] += 1;
                // cout << " code run here 67? " << endl;
        //   //     } 
        //   //   }
        //   // }        
          // cout << " code run here 2 ? " << endl;

        //   // sell the stocks and add money
        //   // if (!holding_stocks.empty()) {        
            // cout << " code run here 68? " << endl;
        //   //   for (std::vector<double>::size_type j = 0; j < stocks_to_sell.size(); j++) {
              // cout << " code run here 69? " << endl;
        //   //     std::vector<IntegerT>::iterator it = std::find(holding_stocks.begin(), holding_stocks.end(), stocks_to_sell[j]);
        //   //     int index = std::distance(holding_stocks.begin(), it);
        //   //     // std::vector<double>::iterator it_shares = std::find(holding_stocks_shares.begin(), holding_stocks_shares.end(), holding_stocks_shares[index]);

        //   //     // std::vector<IntegerT>::iterator it_days = std::find(holding_days.begin(), holding_days.end(), holding_days[index]);          
        //   //     if (it != holding_stocks.end()) {
                // cout << " code run here 70? " << endl;
        //   //       holding_stocks.erase(it);

        //   //       /// James: '''change from holding shares to just holding money for simplicity. The difference with holding shares is just minimum hands of shares to hold (e.g. 100 shares or integer shares)''';
        //   //       // init_cap += holding_stocks_shares[index] * prices_column[stocks_to_sell[j]];
        //   //       init_cap += holding_stocks_shares[index];

        //   //       holding_stocks_shares.erase(holding_stocks_shares.begin()+index);

        //   //       holding_counts += 1;
        //   //       holding_days_total += holding_days[index];

        //   //       holding_days.erase(holding_days.begin()+index);
        //   //     } else {
        //   //       std::cout << "why cannot find stocks_to_sell in holding_stocks??" << std::endl;
        //   //     }
        //   //   }
        //   // }
        // }

        // buy the stocks and add shares
        // std::cout << "NAV/2 " << NAV/2 << std::endl;
        // std::cout << "new_balance " << new_balance<< std::endl;
        // std::cout << "init_cap " << init_cap << std::endl;
        // init_cap -= new_balance - (NAV/2);
        // cash += new_balance - (NAV/2);

        if (stocks_to_buy.size() != 0) {
          CHECK(stocks_to_buy.size() == num_holding);
          // CHECK(holding_stocks.size() < num_holding);
          double cash_per_stock = (NAV/2) / stocks_to_buy.size();
          for (std::vector<double>::size_type j = 0; j < stocks_to_buy.size(); j++) {
            /// James: '''change from holding shares to just holding money for simplicity. The difference with holding shares is just minimum hands of shares to hold (e.g. 100 shares or integer shares)''';
            // holding_stocks_shares.push_back(cash_per_stock / prices_column[stocks_to_buy[j]]);

            if (std::find(holding_stocks.begin(), holding_stocks.end(), stocks_to_buy[j]) != holding_stocks.end()) {
              std::vector<IntegerT>::iterator it = std::find(holding_stocks.begin(), holding_stocks.end(), stocks_to_buy[j]);
              int index = std::distance(holding_stocks.begin(), it);
              // std::vector<double>::iterator it_shares = std::find(holding_stocks_shares.begin(), holding_stocks_shares.end(), holding_stocks_shares[index]);
              // cout << " code run here 4? " << endl;
              // std::vector<IntegerT>::iterator it_days = std::find(holding_days.begin(), holding_days.end(), holding_days[index]);          
              holding_stocks_shares[index] = cash_per_stock;
              holding_days[index] += 1;                     
            } else {
              holding_stocks_shares.push_back(cash_per_stock);
              // cout << "holding stock shares[" << j << "]: " << cash_per_stock << endl;
              if (cash_per_stock > 1000000) {
                std::cout << "init_cap " << init_cap << std::endl;
                std::cout << "stocks_to_buy.size() " << stocks_to_buy.size() << std::endl;
                CHECK(cash_per_stock < 1000000);
              }
              holding_stocks.push_back(stocks_to_buy[j]);
              // cout << "holding stock[" << j << "]: " << stocks_to_buy[j] << endl;
              holding_days.push_back(1);
            }       
          }
        }
        // init_cap = 0;
        CHECK(holding_stocks.size() == num_holding);
        CHECK(holding_stocks_shares.size() == num_holding);
        CHECK(holding_days.size() == num_holding);
        // std::cout << "holding_stocks.size() 2 " << holding_stocks.size() << std::endl;

        // buy the stocks and add shares
        // std::cout << "NAV/2 " << NAV/2 << std::endl;
        // std::cout << "new_balance_short " << new_balance_short << std::endl;
        // std::cout << "init_cap_short " << init_cap_short << std::endl;
        // init_cap_short += new_balance_short - (NAV/2);
        // cash -= new_balance_short - (NAV/2);
        // std::cout << "holding_stocks_short.size() " << holding_stocks_short.size() << std::endl;
        // if (holding_stocks_short.size() != 0) {
        //   std::cout << "holding_stocks_short[0] " << holding_stocks_short[0] << std::endl;
        //   }        
        if (stocks_to_short.size() != 0) {
          CHECK(stocks_to_short.size() == num_holding);
          // CHECK(holding_stocks_short.size() < num_holding); /// james: comment off because it seems thare are exact same predictions...
          double cash_per_stock_short = (NAV/2) / stocks_to_short.size();
          for (std::vector<double>::size_type j = 0; j < stocks_to_short.size(); j++) {
            /// James: '''change from holding shares to just holding money for simplicity. The difference with holding shares is just minimum hands of shares to hold (e.g. 100 shares or integer shares)''';
            // holding_stocks_shares.push_back(cash_per_stock / prices_column[stocks_to_buy[j]]);
            // if (holding_stocks_short.begin() == holding_stocks_short.end()) {
            //   std::cout << "this is equal!!!!!! " << std::endl;
            // }
            if (std::find(holding_stocks_short.begin(), holding_stocks_short.end(), stocks_to_short[j]) != holding_stocks_short.end()) {
              std::vector<IntegerT>::iterator it = std::find(holding_stocks_short.begin(), holding_stocks_short.end(), stocks_to_short[j]);
              int index = std::distance(holding_stocks_short.begin(), it);
              // std::vector<double>::iterator it_shares = std::find(holding_stocks_shares.begin(), holding_stocks_shares.end(), holding_stocks_shares[index]);
              // cout << " code run here 4? " << endl;
              // std::vector<IntegerT>::iterator it_days = std::find(holding_days.begin(), holding_days.end(), holding_days[index]);          
              holding_stocks_shares_short[index] = cash_per_stock_short;
              holding_days_short[index] += 1;                     
            } else {
              holding_stocks_shares_short.push_back(cash_per_stock_short);
              // cout << "holding stock shares[" << j << "]: " << cash_per_stock_short << endl;
              // if (cash_per_stock > 1000000) {
              //   std::cout << "init_cap " << init_cap << std::endl;
              //   std::cout << "stocks_to_buy.size() " << stocks_to_buy.size() << std::endl;
              //   CHECK(cash_per_stock < 1000000);
              // }
              holding_stocks_short.push_back(stocks_to_short[j]);
              // cout << "holding stock[" << j << "]: " << stocks_to_short[j] << endl;
              holding_days_short.push_back(1);
            }       
          }            
        }

        cash += new_balance - (NAV/2);
        cash -= new_balance_short - (NAV/2);
       
        // std::cout << "holding_stocks_short.size() " << holding_stocks_short.size() << std::endl;
        CHECK(holding_stocks_short.size() == num_holding);
        CHECK(holding_stocks_shares_short.size() == num_holding);
        CHECK(holding_days_short.size() == num_holding);
        // std::cout << "holding_stocks.size() 2 " << holding_stocks.size() << std::endl;
        // }
        // Compute the strategy return
        double strategy_return = 0.0;
        new_balance = 0.0; // james: if use definition then new_balance used before this declaration will follow initialization from outside the loop!!!
        double strategy_return_short = 0.0;
        new_balance_short = 0.0;
        double overall_strategy_return = 0.0;
        for (std::vector<double>::size_type j = 0; j < holding_stocks.size(); j++) {
          new_balance += holding_stocks_shares[j] * (price_diff_column[holding_stocks[j]] + 1);
          // std::cout << "new_balance " << new_balance << std::endl;
          // std::cout << "holding_stocks_shares[j] " << holding_stocks_shares[j] << std::endl;
          // if (holding_stocks_shares[j] > 100000) CHECK(holding_stocks_shares[j] < 100000);
          // std::cout << "prices_column[holding_stocks[j]] " << prices_column[holding_stocks[j]] << std::endl;
          // std::cout << "price_diff_column[holding_stocks[j]] + 1 " << price_diff_column[holding_stocks[j]] + 1 << std::endl;
        }
        for (std::vector<double>::size_type j = 0; j < holding_stocks_short.size(); j++) {
          new_balance_short += holding_stocks_shares_short[j] * (price_diff_column[holding_stocks_short[j]] + 1);
        }
        // std::cout << "new_balance " << new_balance << std::endl;
        // std::cout << "i: " << i << std::endl;
        NAV = new_balance + cash - new_balance_short;
        strategy_return = new_balance/old_balance - 1;
        strategy_return_short = 1 - new_balance_short/old_balance_short;
        // std::cout << "cash " << cash << std::endl;
        // std::cout << "new_balance " << new_balance << std::endl;
        // std::cout << "strategy_return " << strategy_return << std::endl;
        // std::cout << "strategy_return_short " << strategy_return_short << std::endl;
        // std::cout << "new_balance_short " << new_balance_short << std::endl;
        // std::cout << "old_balance_short " << old_balance_short << std::endl;
        overall_strategy_return = (strategy_return_short + strategy_return)/2; /// james: keep a total leverage of 1
        overall_strategy_return = NAV / old_NAV - 1;
        old_NAV = NAV;
        // std::cout << "overall_strategy_return " << overall_strategy_return << std::endl;
        strategy_returns.push_back(overall_strategy_return);
        if (strategy_ret != nullptr) {
          strategy_ret->push_back(overall_strategy_return);
        }
        // if (overall_strategy_return != 0) {
        //   CHECK(overall_strategy_return == 0);
        // } else {
        //   std::cout << "overall_strategy_return == 0 " << std::endl;
        // }
        old_balance = new_balance;
        old_balance_short = new_balance_short;

        if (isnan(new_balance)) CHECK(!isnan(new_balance));
        if (isnan(new_balance_short)) CHECK(!isnan(new_balance_short));

        // Compute the max dropdown
        if (overall_strategy_return < 0) {
          dropdown_strat_ret_accu *= (overall_strategy_return + 1);
          if (lowest_dropdown_strat_ret_accu > dropdown_strat_ret_accu) lowest_dropdown_strat_ret_accu = dropdown_strat_ret_accu;
        } else {
          dropdown_strat_ret_accu = 1.0;
        } 
        CHECK(preds_column.size() == price_diff_column.size());
      }
      // std::cout << "code 24 " << std::endl;
      *strat_ret_vol = stdev(strategy_returns);
      *annual_mean_strat_ret = mean(strategy_returns) * 252;
      // std::cout << "code 23 " << std::endl;
      if (stdev(strategy_returns) == 0 || std::abs(mean(strategy_returns)) < 0.000000001) { /// james: second condition add because some sharpe ratio is out without valid prediction is due to rounding error, e.g. e-15 / e-16 give a good sharpe ratio
        *sharpe_ratio = 0;
      } else {
        *sharpe_ratio = mean(strategy_returns) / stdev(strategy_returns) * sqrt(252);
        // std::cout << "mean(strategy_returns)" << mean(strategy_returns) << std::endl;
        // std::cout << "stdev(strategy_returns)" << stdev(strategy_returns) << std::endl;
        // std::cout << "mean(strategy_returns) / stdev(strategy_returns) * sqrt(252)" << mean(strategy_returns) / stdev(strategy_returns) * sqrt(252) << std::endl;
      }
      // std::cout << "mean(strategy_returns)" << mean(strategy_returns) << std::endl;
      // std::cout << "stdev(strategy_returns)" << stdev(strategy_returns) << std::endl;
      // std::cout << "mean(strategy_returns) / stdev(strategy_returns) * sqrt(252)" << mean(strategy_returns) / stdev(strategy_returns) * sqrt(252) << std::endl;
      if (holding_counts == 0) {
        *average_holding_days = 0;
      } else {
        *average_holding_days = holding_days_total / holding_counts;
      }
      // std::cout << "code 22 " << std::endl;
      // if (lowest_dropdown_strat_ret_accu == 1.0) {
      //  for (std::vector<double>::size_type j = 0; j < strategy_returns.size(); j++) {
      //     std::cout << "strategy_returns[" << j << "]: " << strategy_returns[j] << std::endl;
      //   }
      // }




    return lowest_dropdown_strat_ret_accu;
}

double Evaluator::CorrelationVec(const std::vector<double> all_task_preds, const std::vector<double> price_diff) {

  IntegerT length = all_task_preds.size();
  if (all_task_preds.size() != price_diff.size()) length = std::min(all_task_preds.size(), price_diff.size());

  vector<double>::const_iterator first_all_task_preds = all_task_preds.begin() + (all_task_preds.size() - length);
  vector<double>::const_iterator last_all_task_preds = all_task_preds.end();
  vector<double> new_all_task_preds(first_all_task_preds, last_all_task_preds);

  vector<double>::const_iterator first_price_diff = price_diff.begin() + (price_diff.size() - length);
  vector<double>::const_iterator last_price_diff = price_diff.end();
  vector<double> new_price_diff(first_price_diff, last_price_diff);

  double sum = 0;
  for (std::vector<double>::size_type j = 0; j < new_all_task_preds.size(); j++) {
    sum += ((new_all_task_preds[j] - mean(new_all_task_preds)) * (new_price_diff[j] - mean(new_price_diff)));
  }

  sum = sum / (stdev(new_all_task_preds) * stdev(new_price_diff) * static_cast<double>(new_all_task_preds.size()));          

  return sum;
}

double Evaluator::Correlation(const std::vector<std::vector<double> > all_task_preds, const std::vector<std::vector<double> > price_diff) {

    CHECK(all_task_preds.size() == price_diff.size());
    for (std::vector<double>::size_type j = 0; j < all_task_preds.size(); j++) {
      CHECK(all_task_preds[j].size() == price_diff[j].size());
    }
    std::vector<double> preds_column;
    std::vector<double> price_diff_column;
    double result = 0;
    for (std::vector<double>::size_type i = 0; i < all_task_preds[0].size(); i++) 
      {      
        // cout << " i: " << i << endl;
        preds_column.clear();
        price_diff_column.clear();
        CHECK(preds_column.size() == 0);
        CHECK(price_diff_column.size() == 0);
        for (std::vector<double>::size_type j = 0; j < all_task_preds.size(); j++) {
            // cout << " i " << i << endl;
            // cout << " j " << j << endl;
          // if (i == 0 && j == 22) {
          //   cout << " data[21][0] " << data[21][i] << endl;
          //   cout << " data[22][0] " << data[j][i] << endl;
          // }
          if ((std::abs(all_task_preds[j][i]) != 1234 && std::abs(price_diff[j][i]) == 1234) || (std::abs(all_task_preds[j][i]) == 1234 && std::abs(price_diff[j][i]) != 1234)) {
            cout << " why not together? " << preds_column[j] << endl;
            cout << " why not together? j+1 " << preds_column[j+1] << endl;
            cout << " j: " << j << endl;
            cout << " i: " << i << endl;
            for (std::vector<double>::size_type k = 0; k < all_task_preds.size(); k++) {
              cout << " price_diff[k][i] " << price_diff[k][i] << endl;
              cout << " preds_column[k] " << preds_column[k] << endl;
              if (k == 30) break;
            }
            cout << " all_task_preds[j-1][i] " << all_task_preds[j-1][i] << endl;
            cout << " price_diff[j-1][i] " << price_diff[j-1][i] << endl;
            cout << " all_task_preds[j][i] " << all_task_preds[j][i] << endl;
            cout << " price_diff[j][i] " << price_diff[j][i] << endl;
            cout << " all_task_preds[j+1][i] " << all_task_preds[j+1][i] << endl;
            cout << " price_diff[j+1][i] " << price_diff[j+1][i] << endl;
            cout << " all_task_preds[j+2][i] " << all_task_preds[j+2][i] << endl;
            cout << " price_diff[j+2][i] " << price_diff[j+2][i] << endl;
            cout << " all_task_preds[j][i-1] " << all_task_preds[j][i-1] << endl;
            cout << " price_diff[j][i-1] " << price_diff[j][i-1] << endl;
            cout << " all_task_preds[j][i+1] " << all_task_preds[j][i+1] << endl;
            cout << " price_diff[j][i+1] " << price_diff[j][i+1] << endl;
            cout << " all_task_preds[j][i+2] " << all_task_preds[j][i+2] << endl;
            cout << " price_diff[j][i+2] " << price_diff[j][i+2] << endl;
            // // write to file
            // std::ofstream outFile("/hdd1/james/google-research/automl_zero/train_my_file_1.txt");
            //   for (auto i = all_task_preds.begin(); i != all_task_preds.end(); ++i) {
            //   for (auto j = (*i).begin(); j != (*i).end(); ++j) {outFile << *j << " ";}
            //   outFile << "\n";
            // }
            // std::ofstream outFileDiff("/hdd1/james/google-research/automl_zero/train_my_file_diff.txt");
            //   for (auto i = price_diff.begin(); i != price_diff.end(); ++i) {
            //   for (auto j = (*i).begin(); j != (*i).end(); ++j) {outFileDiff << *j << " ";}
            //   outFileDiff << "\n";
            // }
            CHECK(j == 1234);
          }
          if (std::abs(all_task_preds[j][i]) != 1234 && std::abs(price_diff[j][i]) != 1234) {
            preds_column.push_back(all_task_preds[j][i]);
            price_diff_column.push_back(price_diff[j][i]);
            // cout << " price_diff[j][i]: " << price_diff[j][i] << endl;
            // CHECK(i != 2);
            if (std::abs(all_task_preds[j][i] * price_diff[j][i]) > 1) {
              cout << " why larger than 1? " << all_task_preds[j][i] << endl;
            }
          } 
          // cout << " result[i][j] " << result[i][j] << endl;
          // cout << " data[j][i] " << data[j][i] << endl;
        }
        CHECK(preds_column.size() == price_diff_column.size());
        double sum = 0;
        for (std::vector<double>::size_type j = 0; j < preds_column.size(); j++) {
          sum += ((preds_column[j] - mean(preds_column)) * (price_diff_column[j] - mean(price_diff_column)));
        }
        if (stdev(preds_column) == 0.0 || preds_column.size() == 0 || price_diff_column.size() == 0) { // James: after adding fec check, sometimes preds_column could have very few stocks' results even nothing. Adding this the latter two conditions check then can avoid preds_column equal to nothing. Add the first condition to check if any nonsense predictions.
          // result += 0;
          return 0; // James: if at one time step all preds are the same then must be not learning anything useful then we should return nothing at all;
        } else {
          result += sum / (stdev(preds_column) * stdev(price_diff_column) * static_cast<double>(preds_column.size()));          
        }
        // cout << "sum / (stdev(preds_column) * stdev(price_diff_column) * all_task_preds.size())" << sum / (stdev(preds_column) * stdev(price_diff_column) * all_task_preds.size()) << endl;
        // if (result == 0.0) {
        //   cout << "all_task_preds" << endl;
        //   for (std::vector<double>::size_type i = 0; i < all_task_preds[0].size(); i++) 
        //     {      
        //       for (std::vector<double>::size_type j = 0; j < all_task_preds.size(); j++) {
        //         std::cout << all_task_preds[j][i] << ' ' << endl;
        //       }
        //     }
        //   // for (auto i = preds_column.begin(); i != preds_column.end(); ++i)
        //   //   std::cout << *i << ' ';
        //   // cout << "price_diff_column" << endl;
        //   // for (auto i = price_diff_column.begin(); i != price_diff_column.end(); ++i)
        //   //   std::cout << *i << ' ';
        // }

        if (isnan(result)) {
          cout << "sum: " << sum << endl;
          cout << "nan preds_stock_0" << endl;
          for (auto i = all_task_preds[0].begin(); i != all_task_preds[0].end(); ++i)
            std::cout << *i << ' ';
          cout << "nan preds_stock_1" << endl;
          for (auto i = all_task_preds[1].begin(); i != all_task_preds[1].end(); ++i)
            std::cout << *i << ' ';
          cout << "nan preds_column" << endl;
          for (auto i = preds_column.begin(); i != preds_column.end(); ++i)
            std::cout << *i << ' ';
          cout << "nan price_diff_column" << endl;
          for (auto i = price_diff_column.begin(); i != price_diff_column.end(); ++i)
            std::cout << *i << ' ';
          cout << "stdev(preds_column): " << stdev(preds_column) << endl;
          cout << "stdev(price_diff_column): " << stdev(price_diff_column) << endl;  
          CHECK(!isnan(result));                  
        }
      }
      result = result/static_cast<double>(all_task_preds[0].size());
      // cout << "IC" << result/((all_task_preds[0].size()) * (all_task_preds.size())) << endl;
      // if (result < -0.03) {
      //   cout << "all_task_preds[-1].size" << all_task_preds[-1].size() << endl;
      //   cout << "price_diff[-1].size" << price_diff[-1].size() << endl;
      //   cout << "all_task_preds[-2].size" << all_task_preds[-2].size() << endl;
      //   cout << "price_diff[-2].size" << price_diff[-2].size() << endl;
      //   cout << "all_task_preds[-3].size" << all_task_preds[-3].size() << endl;
      //   cout << "price_diff[-3].size" << price_diff[-3].size() << endl;
      //   cout << "all_task_preds[-4].size" << all_task_preds[-4].size() << endl;
      //   cout << "price_diff[-4].size" << price_diff[-4].size() << endl;
      //   cout << "all_task_preds[-5].size" << all_task_preds[-5].size() << endl;
      //   cout << "all_task_preds.size()" << all_task_preds.size() << endl;
      //   cout << "price_diff.size()" << price_diff.size() << endl;
      //   for (std::vector<double>::size_type j = 0; j < all_task_preds.size(); j++) {
      //     cout << "j" << j << endl;
      //     if (all_task_preds[j].size() == 0) {
      //       cout << "all_task_preds[j].size() == 0" << endl;
      //     }
      //     if (price_diff[j].size() == 0) {
      //       cout << "price_diff[j].size() == 0" << endl;
      //     }
      //     cout << "all_task_preds[j].size" << all_task_preds[j].size() << endl;
      //     cout << "price_diff[j].size()" << price_diff[j].size() << endl;
      //   }
      //   for (auto i = all_task_preds.begin(); i != all_task_preds.end(); ++i) {
      //     if (i->size() != 244) {
      //       cout << "all_task_preds[i].size" << i->size() << endl;
      //     }
      //   }
      //   for (auto i = price_diff.begin(); i != price_diff.end(); ++i) {
      //     if (i->size() != 244) {
      //       cout << "price_diff[i].size" << i->size() << endl;
      //     }
      //   }
      //   // write to file
      //   IntegerT count = 0;
      //   std::ofstream outFile("/hdd1/james/google-research/automl_zero/train_my_file_negative_debug_new.txt");
      //     for (auto i = all_task_preds.begin(); i != all_task_preds.end(); ++i) {
      //     for (auto j = (*i).begin(); j != (*i).end(); ++j) {
      //       outFile << *j << " ";
      //       if (count == 822 || count == 823 || count == 824) {
      //         cout << "i" << count << endl;
      //         cout << *j << endl; 
      //       }
      //     }

      //     outFile << "\n";
      //     ++count;
      //   }
      //   outFile.close();
      //   std::ofstream outFileDiff("/hdd1/james/google-research/automl_zero/train_my_file_diff_negative_debug_new.txt");
      //     for (auto i = price_diff.begin(); i != price_diff.end(); ++i) {
      //     for (auto j = (*i).begin(); j != (*i).end(); ++j) {outFileDiff << *j << " ";}
      //     outFileDiff << "\n";
      //   }
      //   outFileDiff.close();
      //   CHECK(result == 1234);
      // }
    return result;
}

double Evaluator::mean(const std::vector<double> v) {
  double sum = std::accumulate(v.begin(), v.end(), 0.0);
  double mean = sum / v.size();
  return mean;
}

double Evaluator::stdev(const std::vector<double> v) {
  double sum = std::accumulate(v.begin(), v.end(), 0.0);
  double mean = sum / v.size();

  std::vector<double> diff(v.size());
  std::transform(v.begin(), v.end(), diff.begin(), [mean](double x) { return x - mean; });
  double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
  double stdev = std::sqrt(sq_sum / v.size());
  return stdev;
}

double Evaluator::Execute(const TaskInterface& task,
                          const IntegerT num_train_examples,
                          const Algorithm& algorithm,
                          std::vector<std::vector<double>> *all_task_preds,
                          std::vector<std::vector<double>> *all_price_diff,
                          std::vector<std::vector<std::vector<double>>>* tasks_rank,
                          IntegerT this_round,
                          IntegerT task_index,
                          IntegerT* num_stock_rank, IntegerT* num_TS_rank, const IntegerT num_of_stocks_to_approximate_rank, const IntegerT all_rounds, std::vector<IntegerT> *useful_list) {
  switch (task.FeaturesSize()) {
    case 2: {
      const Task<2>& downcasted_task = *SafeDowncast<2>(&task); // James: down is how to change the task/dataset? Later in execute task<F> is treated as dataset
      return ExecuteImpl<2>(downcasted_task, num_train_examples, algorithm, all_task_preds, all_price_diff, tasks_rank, this_round, task_index, num_stock_rank, num_TS_rank, num_of_stocks_to_approximate_rank, all_rounds, useful_list);
    }
    case 4: {
      const Task<4>& downcasted_task = *SafeDowncast<4>(&task);
      return ExecuteImpl<4>(downcasted_task, num_train_examples, algorithm, all_task_preds, all_price_diff, tasks_rank, this_round, task_index, num_stock_rank, num_TS_rank, num_of_stocks_to_approximate_rank, all_rounds, useful_list);
    }
    case 8: {
      const Task<8>& downcasted_task = *SafeDowncast<8>(&task);
      return ExecuteImpl<8>(downcasted_task, num_train_examples, algorithm, all_task_preds, all_price_diff, tasks_rank, this_round, task_index, num_stock_rank, num_TS_rank, num_of_stocks_to_approximate_rank, all_rounds, useful_list);
    }
    case 10: {
      const Task<10>& downcasted_task = *SafeDowncast<10>(&task);
      return ExecuteImpl<10>(downcasted_task, num_train_examples, algorithm, all_task_preds, all_price_diff, tasks_rank, this_round, task_index, num_stock_rank, num_TS_rank, num_of_stocks_to_approximate_rank, all_rounds, useful_list);
    }
    case 13: {
      const Task<13>& downcasted_task = *SafeDowncast<13>(&task);
      // std::cout << "task_index" << task_index << std::endl;
      // std::cout << "downcasted_task.industry_relation_" << downcasted_task.industry_relation_ << std::endl;
      return ExecuteImpl<13>(downcasted_task, num_train_examples, algorithm, all_task_preds, all_price_diff, tasks_rank, this_round, task_index, num_stock_rank, num_TS_rank, num_of_stocks_to_approximate_rank, all_rounds, useful_list);
    }
    case 16: {
      const Task<16>& downcasted_task = *SafeDowncast<16>(&task);
      return ExecuteImpl<16>(downcasted_task, num_train_examples, algorithm, all_task_preds, all_price_diff, tasks_rank, this_round, task_index, num_stock_rank, num_TS_rank, num_of_stocks_to_approximate_rank, all_rounds, useful_list);
    }
    case 32: {
      const Task<32>& downcasted_task = *SafeDowncast<32>(&task);
      return ExecuteImpl<32>(downcasted_task, num_train_examples, algorithm,  all_task_preds, all_price_diff, tasks_rank, this_round, task_index, num_stock_rank, num_TS_rank, num_of_stocks_to_approximate_rank, all_rounds, useful_list);
    }
    default:
      LOG(FATAL) << "Unsupported features size." << endl;
  }
}

IntegerT Evaluator::GetNumTrainStepsCompleted() const {
  return num_train_steps_completed_;
}

template <FeatureIndexT F>
double Evaluator::ExecuteImpl(const Task<F>& task,
                              const IntegerT num_train_examples,
                              const Algorithm& algorithm,
                              std::vector<std::vector<double>>* all_task_preds,
                              std::vector<std::vector<double>>* all_price_diff,
                              std::vector<std::vector<std::vector<double>>>* tasks_rank,
                              IntegerT this_round,
                              IntegerT task_index,
                              IntegerT* num_stock_rank, IntegerT* num_TS_rank, const IntegerT num_of_stocks_to_approximate_rank, const IntegerT all_rounds, std::vector<IntegerT> *useful_list) {
  // IntegerT ins_count_rank = 0; 
  // vector<std::string> previous_out;
  // vector<double> levels;

  // for (const std::shared_ptr<const Instruction>& myinstruction :
  //  algorithm->predict_) {

  //   std::string line = myinstruction->ToString();
  //   std::string out = line.substr(0, line.find(delimiterEq)-1);

  //   if (instruction->op_ == 66 || instruction->op_ == 73) {
      
  //     current_in1 = "s" + std::to_string(instruction->in1_);  // only when selected ops we can capture in1 as such otherwise we don't know it's scalar or vector or matrix
  //     current_in2 = "s" + std::to_string(instruction->in2_);      

  //     if (std::find(previous_out.begin(), previous_out.end(), current_in1) != previous_out.end())
  //     {
  //       levels.push_back(0);
  //     }       
  //   }
    
  //   previous_out.push_back(out);
  
  //   // cout << "check instruction: " << myinstruction->ToString() << endl;
  //   if (ins_count_rank != ins_count) { /// only allow rank of scalar number that is calculated before rank operation and it's on the left hand side of equation otherwise encounter loop over loop...
  //     ++ins_count_rank;
  //     if (myinstruction->out_ == in1) {
  //       // cout << "myinstruction->out_" << myinstruction->out_ << endl;
  //       // cout << "in1" << in1 << endl;
  //       // cout << "ins_count_rank" << ins_count_rank << endl;
  //       // cout << "ins_count" << ins_count << endl;

  //       return true;
  //     }
  //   } else return false;
  // }


  Executor<F> executor(
      algorithm, task, num_train_examples, task.ValidSteps(),
      rand_gen_, max_abs_error_);
  vector<double> valid_preds;
  vector<double> price_diff;
  const double fitness = executor.Execute(&valid_preds, &price_diff, tasks_rank, this_round, task_index, num_stock_rank, num_TS_rank, num_of_stocks_to_approximate_rank, nullptr, nullptr, useful_list);
  // James: sometimes price_diff can be empty since executor.Execute can be return early is executor.train is interrupted by nan pred. 
  // This early return will make the valid preds empty vector since I only fill preds in the executor.Validate function which comes after 
  // executor.train in executor.train part.
    // cout << "code run here 8 " << endl;
  if (this_round == all_rounds - 1) {
    all_task_preds->push_back(valid_preds);
    all_price_diff->push_back(price_diff);      
  }
  // if (valid_preds.empty()) {
  //   cout << " valid_preds.size() " << valid_preds.size() << endl;
  // }
  // if (valid_preds.size() == 0) cout << " valid_preds.size() " << valid_preds.size() << endl;
    // cout << "code run here 9 " << endl;
  // CHECK_GT(valid_preds.size(), 0);
  // cout << " valid_preds.size() " << valid_preds.size() << endl;
  num_train_steps_completed_ += executor.GetNumTrainStepsCompleted();
  return fitness;
  
}

namespace internal {

double Median(vector<double> values) {  // Intentional copy.
  const size_t half_num_values = values.size() / 2;
  nth_element(values.begin(), values.begin() + half_num_values, values.end());
  return values[half_num_values];
}

double CombineFitnesses(
    const vector<double>& task_fitnesses,
    const FitnessCombinationMode mode) {
  if (mode == MEAN_FITNESS_COMBINATION) {
    double combined_fitness = 0.0;
    for (const double fitness : task_fitnesses) {
      combined_fitness += fitness;
    }
    combined_fitness /= static_cast<double>(task_fitnesses.size());
    return combined_fitness;
  } else if (mode == MEDIAN_FITNESS_COMBINATION) {
    return Median(task_fitnesses);
  } else {
    LOG(FATAL) << "Unsupported fitness combination." << endl;
  }
}

}  // namespace internal

}  // namespace automl_zero
