#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <likwid-marker.h>

double *function_a(const double *A, const double *x, const int N) {
  LIKWID_MARKER_START("function_a");
  double *y = new double[N];
  for (unsigned int i = 0; i < N; i++) {
    y[i] = 0;
  }
  for (unsigned int i = 0; i < N; i++) {
    for (unsigned int j = 0; j < N; j++) {
      y[i] += A[i * N + j] * x[j];
    }
  }
  LIKWID_MARKER_STOP("function_a");
  return y;
}

double *function_b(const double a, const double *u, const double *v, const int N) {
  double *x = new double[N];
  for (unsigned int i = 0; i < N; i++) {
    x[i] = a * u[i] + v[i];
  }
  return x;
}

double *function_c(const double s, const double *x, const double *y, const int N) {
  double *z = new double[N];
  for (unsigned int i = 0; i < N; i++) {
    if (i % 2 == 0) {
      z[i] = s * x[i] + y[i];
    } else {
      z[i] = x[i] + y[i];
    }
  }
  return z;
}

double function_d(const double *u, const double *v, const int N) {
  double s = 0;
  for (unsigned int i = 0; i < N; i++) {
    s += u[i] * v[i];
  }
  return s;
}

void init_datastructures(double *u, double *v, double *A, const int N) {
  for (unsigned int i = 0; i < N; i++) {
    u[i] = static_cast<double>(i%2);
    v[i] = static_cast<double>(i%4);
  }

  for (unsigned int i = 0; i < N * N; i++) {
    A[i] = static_cast<double>(i%8);
  }
}

void print_results_to_file(const double s, const double *x, const double *y,
                           const double *z, const double *A, const long long n,
                           std::ofstream &File) {
  unsigned int N = std::min(n, static_cast<long long>(30));

  File << "N: "
       << "\n"
       << n << "\n";

  File << "s: "
       << std::fixed
       << std::setprecision(1)
       << "\n"
       << s << "\n";

  File << "x: "
       << "\n";
  for (unsigned int i = 0; i < N; i++) {
    File << x[i] << " ";
  }
  File << "\n";

  File << "y: "
       << "\n";
  for (unsigned int i = 0; i < N; i++) {
    File << y[i] << " ";
  }
  File << "\n";

  File << "z: "
       << "\n";
  for (unsigned int i = 0; i < N; i++) {
    File << z[i] << " ";
  }
  File << "\n";
}

int main(int argc, char **argv) {
  LIKWID_MARKER_INIT;
  long long N = 10000;

  double *u = new double[N];
  double *v = new double[N];
  double *A = new double[N * N];

  init_datastructures(u, v, A, N);

  double s = function_d(u, v, N);
  double *x = function_b(2, u, v, N);
  double *y = function_a(A, x, N);
  double *z = function_c(s, x, y, N);

  std::cout << "Profiling complete, results saved." << std::endl;

  delete[] u;
  delete[] v;
  delete[] A;
  delete[] x;
  delete[] y;
  delete[] z;

  LIKWID_MARKER_CLOSE;
  
  return 0;
}
