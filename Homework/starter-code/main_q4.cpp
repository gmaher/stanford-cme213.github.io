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

	std::vector<T> v;

	std::for_each(data.begin(), data.end(),
		[&a,&y,&v](T x){v.push_back(a*x+y);});

	return v;
}

template <typename T>
void daxpy(std::vector<T>& data, T a, T y) {

	std::transform(data.begin(), data.end(), data.begin(),
		[a,y](T x){return a*x+y;});

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
	return std::all_of(students.begin(), students.end(),
			[pass_threshold](Student s){
				double score = HOMEWORK_WEIGHT*s.homework+MIDTERM_WEIGHT*s.midterm+
					FINAL_EXAM_WEIGHT*s.final_exam;
				return score >= pass_threshold;
			});
}


/**********  Q4c: Odd first, even last **********/
void sort_odd_even(std::vector<int>& data) {
	std::sort(data.begin(), data.end());

	std::vector<int> odd, even;
	std::for_each(data.begin(), data.end(),
		[&odd](int x){ if(x%2){odd.push_back(x);} });

	std::for_each(data.begin(), data.end(),
		[&even](int x){ if(!(x%2)){even.push_back(x);} });

	std::vector<int> result;

	std::for_each(odd.begin(), odd.end(), [&result](int x){result.push_back(x);});
	std::for_each(even.begin(), even.end(), [&result](int x){result.push_back(x);});
	data = result;
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

	//Qa
	const std::vector<int> v({1,2,3});
	int a = 2;
	int y = 3;

	auto vd = daxpy(v,a,y);
	daxpy(vd,a,y);
	for (int i = 0; i < v.size(); i++){
		std::cout << vd[i] << "\n";
	}


	//Qb
	int thresh = 60;
	std::vector<Student> students;
	students.push_back(Student(80,80,80));
	students.push_back(Student(90,90,90));

	std::cout << "All students passed? " << all_students_passed(students, thresh) << "\n";

	students.push_back(Student(10,20,90));
	std::cout << "All students passed? " << all_students_passed(students, thresh) << "\n";

	//Qc
	std::vector<int> numbers({4,5,3,2});
	sort_odd_even(numbers);
	for (int i = 0; i <  numbers.size(); i++){
		std::cout << numbers[i] << "\n";
	}
	// TODO: Write your tests here!
	return 0;
}
