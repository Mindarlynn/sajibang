#include <iostream>
#include <complex>
#include <vector>
#include <ctime>

#define TRAIN_SET_SIZE 20
#define PI 3.141592653589793238463

#define N 5

#define epsilon 0.05
#define epoch 50000

double c[N] = {};
double W[N] = {};
double V[N] = {};
double b = 0.;

double sigmoid(double x){
	return 1. / (1. + std::exp(-x));
}

double f_theta(double x){
	double result = b;
	for(int i = 0; i < N; ++i){
		result += V[i] * sigmoid(c[i] + W[i] * x);
	}
	return result;
}

void train(double x, double y){
	for(int i = 0; i < N; ++i){
		W[i] = W[i] - epsilon * 2 * (f_theta(x) - y) * V[i] * x * 
			(1 - sigmoid(c[i] + W[i] * x)) * sigmoid(c[i] + W[i] * x);
	}
	for(int i = 0; i < N; ++i){
		V[i] = V[i] - epsilon * 2 * (f_theta(x) - y) * 
			sigmoid(c[i] + W[i] * x);
	}
	b = b - epsilon * 2 * (f_theta(x) - y);
	for(int i = 0; i < N; ++i){
		c[i] = c[i] - epsilon * 2 * (f_theta(x) - y) * V[i] * 
			(1 - sigmoid(c[i] + W[i] * x)) * sigmoid(c[i] + W[i] * x);
	}
}

int main(){
	srand(time(0));
	for(int i = 0; i < N; ++i){
		W[i] = 2 * rand() / RAND_MAX - 1;
		V[i] = 2 * rand() / RAND_MAX - 1;
		c[i] = 2 * rand() / RAND_MAX - 1;
	}
	
	std::vector<std::pair<double, double>> train_set{};
	train_set.resize(TRAIN_SET_SIZE);
	
	for(int i = 0; i < TRAIN_SET_SIZE; ++i){
		train_set[i] = std::make_pair(i * 2 * PI / TRAIN_SET_SIZE, sin(i * 2 * PI / TRAIN_SET_SIZE));
	}
	
	for(int j = 0; j < epoch; ++j){
		for(int i = 0; i < TRAIN_SET_SIZE; ++i){
			train(train_set[i].first, train_set[i].second);
		}
		std::cout<<j + 1 << " / " << epoch << std::endl;
	}
	
	std::vector<double> x;
	std::vector<double> y1, y2;
	
	for(int i = 0; i < 1000; ++i){
		x.push_back(i * 2 * PI / 1000);
		y1.push_back(sin(i * 2 * PI / 1000));
		y2.push_back(f_theta(i * 2 * PI / 1000));
	}
	
	auto file = fopen("plot", "w");
	
	fprintf(file, "f(x) = sin(x)\n");
	for(int i = 0; i < 1000; ++i){
		fprintf(file, "%f %f\n", x[i], y1[i]);
	}
	fprintf(file, "f(x) = sin(x) w/ neural network\n");
	for(int i = 0; i < 1000; ++i){
		fprintf(file, "%f %f\n", x[i], y2[i]);
	}
	
	return 0;
}