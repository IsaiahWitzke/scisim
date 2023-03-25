// Utilities.cpp
//
// Breannan Smith
// Last updated: 09/03/2015

#include "Utilities.h"
#include <string>
#include <iostream>

void Utilities::printSparseMatrixJson(SparseMatrixsc m, std::string name = "")
{
  if (name != "")
  {
    std::cout << "\"" << name << "\": ";
  }
  std::cout << "{" << std::endl;
  std::cout << "\"shape\": [" << m.rows() << ", " << m.cols() << "]," << std::endl;

  std::cout << "\"data\": [";
  for (int k = 0; k < m.outerSize(); ++k)
  {
    for (SparseMatrixsc::InnerIterator it(m, k); it; ++it)
    {
      std::cout.precision(12);
      std::cout << "[" << it.row() << "," << it.col() << "," << std::fixed << double(it.value()) << "], ";
    }
  }
  std::cout << "]" << std::endl;
  std::cout << "}," << std::endl;
}

void Utilities::printVectorJson(VectorXs v, std::string name = "")
{
  if (name != "")
  {
    std::cout << "\"" << name << "\": ";
  }
  std::cout << "{" << std::endl;
  std::cout << "\"shape\": [" << v.rows() << ", " << v.cols() << "]," << std::endl;

  std::cout << "\"data\": [";
  for (int row = 0; row < v.rows(); ++row)
  {
    std::cout.precision(12);
    std::cout << "[" << row << "," << 0 << "," << std::fixed << double(v(row)) << "], ";
  }

  std::cout << "]" << std::endl;
  std::cout << "}," << std::endl;
}

void Utilities::printBoolJson(bool b, std::string name)
{
  std::cout << "\"" << name << "\": ";
  if (b)
  {
    std::cout << "true";
  }
  else
  {
    std::cout << "false";
  }
  std::cout << "," << std::endl;
}

//
// Error statistics of solvers (LCP)
//

double Utilities::getEndError(const SparseMatrixsc &Q, const VectorXs &x, const VectorXs &b)
{
  scalar err2 = 0;
  VectorXs y = Q * x + b; // resultant vector
  for (int i = 0; i < x.size(); ++i)
  {
    err2 += fmin(x(i), y(i)) * fmin(x(i), y(i));
  }
  return sqrt(err2);
}

double Utilities::getAbsDiff(VectorXs a, VectorXs b)
{
  scalar err = 0;
  for (int i = 0; i < a.size(); ++i)
  {
    err += ((a(i) - b(i)) * (a(i) - b(i)));
  }
  return sqrt(err);
}

double Utilities::getKineticEnergy(const SparseMatrixsc &M, const VectorXs &v)
{
  return 0.5 * v.transpose() * M * v;
}

template <>
void Utilities::serialize<bool>(const std::vector<bool> &vector, std::ostream &output_stream)
{
  assert(output_stream.good());
  serialize(vector.size(), output_stream);
  for (std::vector<bool>::size_type idx = 0; idx < vector.size(); ++idx)
  {
    const bool local_val = vector[idx];
    serialize(local_val, output_stream);
  }
}

template <>
std::vector<bool> Utilities::deserialize<std::vector<bool>>(std::istream &input_stream)
{
  assert(input_stream.good());
  std::vector<bool> vector;
  const std::vector<bool>::size_type length{deserialize<std::vector<bool>::size_type>(input_stream)};
  vector.resize(length);
  for (std::vector<bool>::size_type idx = 0; idx < length; ++idx)
  {
    vector[idx] = deserialize<bool>(input_stream);
  }
  return vector;
}
