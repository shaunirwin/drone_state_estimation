#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Dense>

using Eigen::MatrixXd;

using namespace std;

int main()
{
  MatrixXd m(2,2);
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = m(1,0) + m(0,1);
  std::cout << m << std::endl;

  vector<string> msg {"Hello", "C++", "World", "from", "VS Code", "and the C++ extension!"};

  for (const string& word : msg)
    {
        cout << word << " ";
    }
    cout << endl;
}