#include <iostream>
#include <algorithm>
#include <vector>
#include <list>
#include <queue>
#include <numeric>
#include <stdexcept>


/**********  Q4a: DAXPY **********/
template <typename T>
std::vector<T> daxpy(const std::vector<T>& data, T a, T y) {
	// TODO
	return std::vector<T>();
}

template <typename T>
void daxpy(std::vector<T>& data, T a, T y) {
	// TODO
}


/**********  Q4b: All students passed **********/
constexpr double HOMEWORK_WEIGHT = 0.20;
constexpr double MIDTERM_WEIGHT = 0.35;
constexpr double FINAL_EXAM_WEIGHT = 0.45;

struct Student {
	double homework;
	double midterm;
	double final_exam;

	Student(double hw, double mt, double fe) : 
    homework(hw), midterm(mt), final_exam(fe) { }
};

bool all_students_passed(const std::vector<Student>& students, double pass_threshold) {
	// TODO 
	return false;
}


/**********  Q4c: Odd first, even last **********/
void sort_odd_even(std::vector<int>& data) {
	// TODO
}

/**********  Q4d: Sparse matrix list sorting **********/
template <typename T>
struct SparseMatrixCoordinate {
	int row;
	int col;
	T data;
	
	SparseMatrixCoordinate(int r, int c, T d) :
    row(r), col(c), data(d) {}
};

template <typename T>
void sparse_matrix_sort(std::list<SparseMatrixCoordinate<T>>& list) {
	// TODO
}


int main() {
	// TODO: Write your tests here!
	return 0;
}
