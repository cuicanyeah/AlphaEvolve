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

#include "regularized_evolution.h"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <ios>
#include <memory>
#include <sstream>
#include <utility>

#include "algorithm.h"
#include "algorithm.pb.h"
#include "task_util.h"
#include "definitions.h"
#include "executor.h"
#include "instruction.h"
#include "random_generator.h"
#include "absl/flags/flag.h"
#include "absl/memory/memory.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace automl_zero {

namespace {

using ::absl::GetCurrentTimeNanos;  // NOLINT
using ::absl::GetFlag;  // NOLINT
using ::absl::make_unique;  // NOLINT
using ::absl::Seconds;  // NOLINT
using ::std::abs;  // NOLINT
using ::std::cout;  // NOLINT
using ::std::endl;  // NOLINT
using ::std::fixed;  // NOLINT
using ::std::make_pair;  // NOLINT
using ::std::make_shared;  // NOLINT
using ::std::pair;  // NOLINT
using ::std::setprecision;  // NOLINT
using ::std::shared_ptr;  // NOLINT
using ::std::unique_ptr;  // NOLINT
using ::std::vector;  // NOLINT
using ::std::numeric_limits;  // NOLINT

constexpr double kLn2 = 0.69314718056;

}  // namespace

RegularizedEvolution::RegularizedEvolution(
    RandomGenerator* rand_gen, const IntegerT population_size,
    const IntegerT tournament_size, const IntegerT progress_every,
    Generator* generator, Evaluator* evaluator, Mutator* mutator, std::string folder_and_name)
    : evaluator_(evaluator),
      rand_gen_(rand_gen),
      start_secs_(GetCurrentTimeNanos() / kNanosPerSecond),
      epoch_secs_(start_secs_),
      epoch_secs_last_progress_(epoch_secs_),
      num_individuals_last_progress_(std::numeric_limits<IntegerT>::min()),
      tournament_size_(tournament_size),
      progress_every_(progress_every),
      initialized_(false),
      generator_(generator),
      mutator_(mutator),
      population_size_(population_size),
      algorithms_(population_size_, make_shared<Algorithm>()),
      fitnesses_(population_size_),
      ICs_(population_size_),      
      num_individuals_(0),
      folder_and_name_(folder_and_name),
      strategy_ret_(strategy_ret_) {}

IntegerT RegularizedEvolution::Init() {
  // Otherwise, initialize the population from scratch.
  const IntegerT start_individuals = num_individuals_;
  std::vector<double> strategy_ret;
  std::vector<double> valid_strategy_ret;
  std::vector<IntegerT> useful_list;
  std::vector<std::pair<double, std::vector<double>>>::iterator fitness_it = fitnesses_.begin();
  double best_train_fitness = 0;
  for (shared_ptr<const Algorithm>& algorithm : algorithms_) {
    InitAlgorithm(&algorithm);
    strategy_ret.clear();
    valid_strategy_ret.clear();
    useful_list.clear();

    *fitness_it = Execute(algorithm, best_train_fitness, &ICs_, &strategy_ret, &valid_strategy_ret, &useful_list);
    ++fitness_it;
  }
  CHECK(fitness_it == fitnesses_.end());
  
  MaybePrintProgress(&strategy_ret, &valid_strategy_ret, &useful_list);
  initialized_ = true;
  return num_individuals_ - start_individuals;
}

IntegerT RegularizedEvolution::Run(const IntegerT max_train_steps,
                                   const IntegerT max_nanos) {
  CHECK(initialized_) << "RegularizedEvolution not initialized."
                      << std::endl;
  std::vector<double> strategy_ret;          
  std::vector<double> valid_strategy_ret; 
  std::vector<IntegerT> useful_list;        
  const IntegerT start_nanos = GetCurrentTimeNanos();
  const IntegerT start_train_steps = evaluator_->GetNumTrainStepsCompleted();
  double best_train_fitness = numeric_limits<double>::lowest();
  while (evaluator_->GetNumTrainStepsCompleted() - start_train_steps <
             max_train_steps &&
         GetCurrentTimeNanos() - start_nanos < max_nanos) {
    vector<std::pair<double, std::vector<double>>>::iterator next_fitness_it = fitnesses_.begin();
    for (shared_ptr<const Algorithm>& next_algorithm : algorithms_) {
      SingleParentSelect(&next_algorithm);
      mutator_->Mutate(1, &next_algorithm);
      strategy_ret.clear();
      valid_strategy_ret.clear();
      useful_list.clear();
      *next_fitness_it = Execute(next_algorithm, best_train_fitness, &ICs_, &strategy_ret, &valid_strategy_ret, &useful_list);
      ++next_fitness_it;
    }
    MaybePrintProgress(&strategy_ret, &valid_strategy_ret, &useful_list);
  }
  return evaluator_->GetNumTrainStepsCompleted() - start_train_steps;
}

IntegerT RegularizedEvolution::NumIndividuals() const {
  return num_individuals_;
}

IntegerT RegularizedEvolution::PopulationSize() const {
  return population_size_;
}

IntegerT RegularizedEvolution::NumTrainSteps() const {
  return evaluator_->GetNumTrainStepsCompleted();
}

shared_ptr<const Algorithm> RegularizedEvolution::Get(
    double* fitness) {
  const IntegerT indiv_index =
      rand_gen_->UniformPopulationSize(population_size_);
  CHECK(fitness != nullptr);
  *fitness = (fitnesses_[indiv_index]).first;
  return algorithms_[indiv_index];
}

shared_ptr<const Algorithm> RegularizedEvolution::GetBest(
    double* fitness) {
  double best_fitness = -1.0;
  IntegerT best_index = -1;
  for (IntegerT index = 0; index < population_size_; ++index) {
    if (best_index == -1 || (fitnesses_[index]).first > best_fitness) {
      best_index = index;
      best_fitness = (fitnesses_[index]).first;
    }
  }
  CHECK_NE(best_index, -1);
  *fitness = best_fitness;
  return algorithms_[best_index];
}

void RegularizedEvolution::PopulationStats(
    double* pop_mean, double* pop_stdev,
    shared_ptr<const Algorithm>* pop_best_algorithm,
    double* pop_best_fitness, double* pop_best_IC, double* pop_best_sharpe_ratio, 
    double* pop_best_correlation, double* pop_best_average_holding_days, double* pop_best_max_dropdown, 
    double* pop_best_strat_ret_vol, double* pop_best_annual_mean_strat_ret, double* pop_best_correlation_with_existing_alpha,
    double* pop_test_sharpe_ratio, double* pop_test_correlation, double* pop_test_average_holding_days, 
    double* pop_test_max_dropdown, double* pop_test_strat_ret_vol, double* pop_test_annual_mean_strat_ret,
    double* pop_test_correlation_with_existing_alpha) const {
  double total = 0.0;
  double total_squares = 0.0;
  double best_fitness = -1.0;
  double best_IC = -1.0;
  IntegerT best_index = -1;
  for (IntegerT index = 0; index < population_size_; ++index) {
    if (best_index == -1 || fitnesses_[index].first > best_fitness) {
      best_index = index;
      best_fitness = fitnesses_[index].first;
      best_IC = ICs_[index];
    }
    const double fitness_double = static_cast<double>(fitnesses_[index].first);
    total += fitness_double;
    total_squares += fitness_double * fitness_double;
  }
  CHECK_NE(best_index, -1);
  double size = static_cast<double>(population_size_);
  const double pop_mean_double = total / size;
  *pop_mean = static_cast<double>(pop_mean_double);
  double var = total_squares / size - pop_mean_double * pop_mean_double;
  if (var < 0.0) var = 0.0;
  *pop_stdev = static_cast<double>(sqrt(var));
  *pop_best_algorithm = algorithms_[best_index];
  *pop_best_fitness = best_fitness;
  *pop_best_IC = best_IC;
  *pop_best_correlation = fitnesses_[best_index].second[0];
  *pop_best_sharpe_ratio = fitnesses_[best_index].second[1];
  *pop_best_average_holding_days = fitnesses_[best_index].second[2];
  *pop_best_max_dropdown = fitnesses_[best_index].second[3];
  *pop_best_strat_ret_vol = fitnesses_[best_index].second[4];
  *pop_best_annual_mean_strat_ret = fitnesses_[best_index].second[5];
  *pop_best_correlation_with_existing_alpha = fitnesses_[best_index].second[6];
  *pop_test_correlation = fitnesses_[best_index].second[7];
  *pop_test_sharpe_ratio = fitnesses_[best_index].second[8];
  *pop_test_average_holding_days = fitnesses_[best_index].second[9];
  *pop_test_max_dropdown = fitnesses_[best_index].second[10];
  *pop_test_strat_ret_vol = fitnesses_[best_index].second[11];
  *pop_test_annual_mean_strat_ret = fitnesses_[best_index].second[12];
  *pop_test_correlation_with_existing_alpha = fitnesses_[best_index].second[13];
}

void RegularizedEvolution::InitAlgorithm(
    shared_ptr<const Algorithm>* algorithm) {
  *algorithm = make_shared<Algorithm>(generator_->TheInitModel());
  // TODO(ereal): remove next line. Affects random number generation.
  mutator_->Mutate(0, algorithm);
}

std::pair<double, std::vector<double>> RegularizedEvolution::Execute(shared_ptr<const Algorithm> algorithm, double best_train_fitness, std::vector<double>* ICs, std::vector<double>* strategy_ret, std::vector<double>* valid_strategy_ret, std::vector<IntegerT>* useful_list) {
  ++num_individuals_;
  epoch_secs_ = GetCurrentTimeNanos() / kNanosPerSecond;
  const IntegerT is_search = 1;
  const std::pair<double, std::vector<double>> fitness = evaluator_->Evaluate(*algorithm, best_train_fitness, is_search, strategy_ret, valid_strategy_ret, useful_list);

  return fitness;
}

shared_ptr<const Algorithm>
    RegularizedEvolution::BestFitnessTournament() {
  double tour_best_fitness = -std::numeric_limits<double>::infinity();
  IntegerT best_index = -1;
  for (IntegerT tour_idx = 0; tour_idx < tournament_size_; ++tour_idx) {
    const IntegerT algorithm_index =
        rand_gen_->UniformPopulationSize(population_size_);
    const double curr_fitness = fitnesses_[algorithm_index].first;
    if (best_index == -1 || curr_fitness > tour_best_fitness) {
      tour_best_fitness = curr_fitness;
      best_index = algorithm_index;
    }
  }
  return algorithms_[best_index];
}

void RegularizedEvolution::SingleParentSelect(
    shared_ptr<const Algorithm>* algorithm) {
  *algorithm = BestFitnessTournament();
}

void RegularizedEvolution::MaybePrintProgress(std::vector<double>* strategy_ret, std::vector<double>* valid_strategy_ret, std::vector<IntegerT>* useful_list) {
  if (num_individuals_ < num_individuals_last_progress_ + progress_every_) {
    return;
  }
  num_individuals_last_progress_ = num_individuals_;
  double pop_mean, pop_stdev, pop_best_fitness, pop_best_IC, pop_best_sharpe_ratio, pop_best_correlation, pop_best_average_holding_days, pop_best_max_dropdown, pop_best_strat_ret_vol, pop_best_annual_mean_strat_ret, pop_best_correlation_with_existing_alpha, pop_test_sharpe_ratio, 
    pop_test_correlation, pop_test_average_holding_days, pop_test_max_dropdown, pop_test_strat_ret_vol, pop_test_annual_mean_strat_ret, pop_test_correlation_with_existing_alpha;
  shared_ptr<const Algorithm> pop_best_algorithm;
  PopulationStats(
      &pop_mean, &pop_stdev, &pop_best_algorithm, &pop_best_fitness, &pop_best_IC, &pop_best_sharpe_ratio, &pop_best_correlation, &pop_best_average_holding_days, &pop_best_max_dropdown, &pop_best_strat_ret_vol, &pop_best_annual_mean_strat_ret, &pop_best_correlation_with_existing_alpha, &pop_test_sharpe_ratio, 
    &pop_test_correlation, &pop_test_average_holding_days, &pop_test_max_dropdown, &pop_test_strat_ret_vol, &pop_test_annual_mean_strat_ret, &pop_test_correlation_with_existing_alpha);
  std::cout << "indivs=" << num_individuals_ << ", " << setprecision(0) << fixed
            << "elapsed_secs=" << epoch_secs_ - start_secs_ << ", "
            << "mean=" << setprecision(6) << fixed << pop_mean << ", "
            << "stdev=" << setprecision(6) << fixed << pop_stdev << ", "
            << "best fit=" << setprecision(6) << fixed << pop_best_fitness << ", " 
            << "best IC=" << setprecision(6) << fixed << pop_best_IC << ", "       
            << "correlation= " << setprecision(6) << fixed << pop_best_correlation << ", "
            << "sharpe_ratio= " << setprecision(6) << fixed << pop_best_sharpe_ratio << ", " 
            << "average_holding_days= " << setprecision(6) << fixed << pop_best_average_holding_days << ", " 
            << "max_dropdown= " << setprecision(6) << fixed << pop_best_max_dropdown << ", " 
            << "strat_ret_vol= " << setprecision(6) << fixed << pop_best_strat_ret_vol << ", " 
            << "annual_mean_strat_ret= " << setprecision(6) << fixed << pop_best_annual_mean_strat_ret << ", "
            << "pop_best_correlation_with_existing_alpha= " << setprecision(6) << fixed << pop_best_correlation_with_existing_alpha << ", "
            << "test correlation= " << setprecision(6) << fixed << pop_test_correlation << ", "
            << "test sharpe_ratio= " << setprecision(6) << fixed << pop_test_sharpe_ratio << ", " 
            << "test average_holding_days= " << setprecision(6) << fixed << pop_test_average_holding_days << ", " 
            << "test max_dropdown= " << setprecision(6) << fixed << pop_test_max_dropdown << ", " 
            << "test strat_ret_vol= " << setprecision(6) << fixed << pop_test_strat_ret_vol << ", " 
            << "test annual_mean_strat_ret= " << setprecision(6) << fixed << pop_test_annual_mean_strat_ret << ", " 
            << "pop_test_correlation_with_existing_alpha= " << setprecision(6) << fixed << pop_test_correlation_with_existing_alpha << endl;
  std::cout.flush();

  std::string filename_perf = "/hdd1/james/google-research/automl_zero/" + folder_and_name_ + "perf" +  std::to_string(num_individuals_) + ".txt";  
  std::string filename = "/hdd1/james/google-research/automl_zero/" + folder_and_name_ + std::to_string(num_individuals_) + ".txt";
  std::string filename_prune = "/hdd1/james/google-research/automl_zero/" + folder_and_name_ + "_pruned_" + std::to_string(num_individuals_) + ".txt";
  std::ofstream outFile(filename);
  outFile << pop_best_algorithm->ToReadable() << " ";
  outFile.close();
  std::ofstream outFileperf(filename_perf);
  outFileperf << "indivs=" << num_individuals_ << ", " << setprecision(0) << fixed
              << "elapsed_secs=" << epoch_secs_ - start_secs_ << ", "
              << "mean=" << setprecision(6) << fixed << pop_mean << ", "
              << "stdev=" << setprecision(6) << fixed << pop_stdev << ", "
              << "best fit=" << setprecision(6) << fixed << pop_best_fitness << ", " 
              << "best IC=" << setprecision(6) << fixed << pop_best_IC << ", "       
              << "correlation= " << setprecision(6) << fixed << pop_best_correlation << ", "
              << "sharpe_ratio= " << setprecision(6) << fixed << pop_best_sharpe_ratio << ", " 
              << "average_holding_days= " << setprecision(6) << fixed << pop_best_average_holding_days << ", " 
              << "max_dropdown= " << setprecision(6) << fixed << pop_best_max_dropdown << ", " 
              << "strat_ret_vol= " << setprecision(6) << fixed << pop_best_strat_ret_vol << ", " 
              << "annual_mean_strat_ret= " << setprecision(6) << fixed << pop_best_annual_mean_strat_ret << ", "
              << "best_correlation_with_existing_alpha= " << setprecision(6) << fixed << pop_best_correlation_with_existing_alpha << ", "
              << "test correlation= " << setprecision(6) << fixed << pop_test_correlation << ", "
              << "test sharpe_ratio= " << setprecision(6) << fixed << pop_test_sharpe_ratio << ", " 
              << "test average_holding_days= " << setprecision(6) << fixed << pop_test_average_holding_days << ", " 
              << "test max_dropdown= " << setprecision(6) << fixed << pop_test_max_dropdown << ", " 
              << "test strat_ret_vol= " << setprecision(6) << fixed << pop_test_strat_ret_vol << ", " 
              << "test annual_mean_strat_ret= " << setprecision(6) << fixed << pop_test_annual_mean_strat_ret << ", "
              << "test_correlation_with_existing_alpha= " << setprecision(6) << fixed << pop_test_correlation_with_existing_alpha << ", strategy_ret: ";
  for (auto i = strategy_ret->begin(); i != strategy_ret->end(); ++i) {
    outFileperf << (*i) << " ";
  }
  outFileperf << ". valid_strategy_ret: ";
  for (auto i = valid_strategy_ret->begin(); i != valid_strategy_ret->end(); ++i) {
    outFileperf << (*i) << " ";
  }
  outFileperf.close();
  
  std::ofstream outFileprune(filename_prune);
  IntegerT ins_countprint = 0;
  for (const std::shared_ptr<const Instruction>& instruction :
       pop_best_algorithm->predict_) {

    if ((*useful_list)[pop_best_algorithm->learn_.size() + ins_countprint] < 1) {
      ++ins_countprint;
      continue;
    } else {
      outFileprune << instruction->ToString() << "\n";
    } 
   ++ins_countprint;
  }
  IntegerT learn_instr_num = 0;
  for (const std::shared_ptr<const Instruction>& instruction :
       pop_best_algorithm->learn_) {
    if ((*useful_list)[learn_instr_num] < 1) {
      ++learn_instr_num;
      continue;
    } else {
      outFileprune << instruction->ToString() << "\n";
    }
    ++learn_instr_num;
  }
  outFileprune.close();

}

}  // namespace automl_zero
