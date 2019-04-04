#include <random>
#include <set>
#include <algorithm>
#include <iostream>

unsigned count_range_entries(std::set<double> S, double lb, double ub){
  std::set<double>::iterator low, up;
  low = S.lower_bound(lb);
  up  = S.upper_bound(ub);

  return std::count_if(low, up, [](double x){return true;});
};

int main(){
  std::set<double> data;
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.0, 1.0);
  for (unsigned int i = 0; i < 1000; ++i) data.insert(distribution(generator));

  std::cout << count_range_entries(data, 2, 10) << " entries in range [2,10]\n";

  return 0;
}
